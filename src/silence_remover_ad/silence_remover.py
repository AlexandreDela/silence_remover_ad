"""
Silence remover
Author: Alexandre Delaisement
"""
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import subprocess # for summoning ffmpeg
import os
import threading
import shutil #for proper check that ffmpeg is installed
from pathlib import Path
from typing import Optional
# Third-party
import numpy as np
from scipy.io import wavfile #for opening wav
import matplotlib.pyplot as plt

class AudioAnalyzer:
    """
    Model class responsible for mathematical analysis of audio data.
    """

    @staticmethod
    def _extract_audio_from_video(video_path: str, audio_output_path: str) -> None:
        videoprocessor = VideoProcessor()
        videoprocessor.extract_audio(video_path, audio_output_path)

    def graph_plot(self, video_file: str, threshold_a: float, threshold_d: float) -> None:
        """
        Plots the graph of amplitude of the audio and of the detected silence

        :param video_file: video file path
        :param threshold_a: amplitude threshold
        :param threshold_d: silence duration threshold
        :return: None
        """
        # Extracting audio out of the video
        path_obj = Path(video_file)
        temp_wav = path_obj.with_suffix('.temp.wav')
        if not temp_wav.exists():
            self._extract_audio_from_video(str(path_obj), str(temp_wav))
        else:
            pass # no need to extract the file again
        wav_path = str(temp_wav)

        # Processing audio
        try:
            sample_rate, data = wavfile.read(wav_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read WAV file using scipy: {e} for path {wav_path}"
            ) from e

        data = self._normalize_data(data)

        # Create a boolean mask where amplitude < threshold_a
        is_silent = data < threshold_a

        # Generate time array
        time = np.arange(len(data)) / sample_rate

        # Plot amplitude and overlay silent regions
        silent_segments = self._determine_silent_segments(is_silent, threshold_d, sample_rate)

        mask = self._calculate_mask(time, silent_segments)

        self._plot(data, time, mask)

    @staticmethod
    def _calculate_mask(time, silent_segments):
        # Calculate mask
        mask = np.zeros_like(time, dtype=bool)
        for start, end in silent_segments:
            mask |= (time >= start) & (time <= end)
        return mask

    @staticmethod
    def _normalize_data(data):
        # Normalize data to float between -1.0 and 1.0 depending on bit depth
        if data.dtype == np.int16:
            data = data / 32768.0
        elif data.dtype == np.int32:
            data = data / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data - 128) / 128.0

        # Handle Stereo (convert to Mono by taking the max absolute value of channels)
        if len(data.shape) > 1:
            data = np.max(np.abs(data), axis=1)
        else:
            data = np.abs(data)
        return data


    @staticmethod
    def _plot(data, time, mask)-> None:
        # Speedup
        speedup_sample_rate = 1000

        plt.figure(figsize=(12, 6))
        plt.plot(
            time[::speedup_sample_rate],
            data[::speedup_sample_rate],
            label='Amplitude',
            color='blue'
        )
        plt.fill_between(
            time[::speedup_sample_rate],
            0,
            np.max(data),
            where=mask[::speedup_sample_rate],
            facecolor='red',
            alpha=0.5,
            label='Silent Regions')
        plt.title('Amplitude and Silent Regions vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()


    def find_silent_segments(
            self,
            wav_path: str,
            threshold_a: float,
            threshold_d: float) \
            -> list[tuple[float, float]] :
        """
        Analyzes a WAV file to find segments where amplitude < threshold_a
        for a duration > threshold_d.

        :param wav_path: Path to the .wav file.
        :param threshold_a: Amplitude threshold (0.0 to 1.0).
        :param threshold_d: Duration threshold in seconds.
        :return list: A list of tuples [(start_time, end_time), ...]
         in seconds representing silence.
        """

        try:
            sample_rate, data = wavfile.read(wav_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read WAV file using scipy: {e}") from e

        # Normalize data to float between -1.0 and 1.0 depending on bit depth
        if data.dtype == np.int16:
            data = data / 32768.0
        elif data.dtype == np.int32:
            data = data / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data - 128) / 128.0

        # Handle Stereo (convert to Mono by taking the max absolute value of channels)
        if len(data.shape) > 1:
            data = np.max(np.abs(data), axis=1)
        else:
            data = np.abs(data)

        # Create a boolean mask where amplitude < threshold_a
        is_silent = data < threshold_a

        # Get a list of tuples of start/end times
        silent_segments: list[tuple[float, float]] = (
            self._determine_silent_segments(is_silent, threshold_d, sample_rate))

        return silent_segments



    @staticmethod
    def _determine_silent_segments(is_silent,
                                   threshold_d: float,
                                   sample_rate: int
                                   ) -> list[tuple[float, float]]:
        """
        Determines silents based on a threshold time.

        :param is_silent: matrix of silence detected
        :param threshold_d: threshold silence duration in seconds
        :param sample_rate: sample rate of the audio file
        :return: List of tuple of start/end of silence, in seconds
        """

        # Find contiguous regions of silence

        # We concatenate False at start/end to detect edges easily (hack bool -> int with np.diff)
        is_silent_padded = np.concatenate(([False], is_silent, [False]))
        diff = np.diff(is_silent_padded.astype(int))

        # Starts are where diff is 1, Ends are where diff is -1
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        silent_segments: list[tuple[float, float]] = []

        # convert threshold seconds -> sample point
        min_samples = int(threshold_d * sample_rate)

        for s, e in zip(starts, ends):
            # Calculate the duration of the silence, in sample time unit (not seconds)
            duration_samples = e - s
            if duration_samples >= min_samples:
                # Convert samples to seconds
                t_start: float = s / sample_rate
                t_end: float = e / sample_rate
                silent_segments.append((t_start, t_end))
        return silent_segments

class VideoProcessor:
    """
    (This class is a façade pattern)
    This class hides the complexity of FFmpeg subprocess calls
    and audio analysis from the rest of the application.
    """

    def __init__(self):
        self.analyzer = AudioAnalyzer()
        self.ffmpeg_cmd = "ffmpeg"  # Assumes ffmpeg is in PATH

    def _check_ffmpeg(self):
        """Verifies that ffmpeg is accessible."""
        if shutil.which(self.ffmpeg_cmd) is None:
            raise FileNotFoundError("FFmpeg is not installed or not found in system PATH.")

    def extract_audio(self, video_path: str, audio_output_path: str, log_callback=print):
        """
        Extract audio to wav using subprocess.

        :param video_path: input video file
        :param audio_output_path: Output audio file
        :param log_callback: Function to send status strings to (for UI updates).
        """
        cmd = [
            self.ffmpeg_cmd, "-y",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # Standard WAV PCM
            "-ar", "44100",  # 44.1kHz
            "-ac", "2",  # Stereo
            audio_output_path
        ]

        log_callback("Extracting the sound file\n" + " ".join(cmd))

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

    @staticmethod
    def _get_video_duration(video_path: str, log_callback=print) -> float:
        """
        Helper to get total video duration using ffprobe.

        :param video_path: Target video
        :param log_callback: Log callback
        :return: Duration in seconds
        """
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]

        log_callback("Probing video duration: \n" + " ".join(cmd))

        result = subprocess.run(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=False)
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0

    def process_video(self, # pylint: disable=R0913,R0917
                      input_path: str,
                      threshold_a: float,
                      threshold_d: float,
                      output_file: Optional[str]=None,
                      log_callback=print
                      ) -> None:
        """
        Execute the full workflow.
        Façade pattern

        :param input_path: Path to source video.
        :param threshold_a: Amplitude threshold.
        :param threshold_d: Duration threshold.
        :param output_file: Optional file path for output video file
        :param log_callback: Function to send status strings to (for UI updates).
        """
        self._check_ffmpeg()

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        path_obj = Path(input_path)
        temp_wav = path_obj.with_suffix('.temp.wav')

        if output_file is None:
            output_video = path_obj.with_name(f"{path_obj.stem}_trimmed{path_obj.suffix}")
        else:
            output_video = path_obj.with_name(output_file)

        try:
            self._extract_audio(path_obj, temp_wav, log_callback)

            # 2. Analyze Audio
            silent_segments = self._analysis_audio(
                temp_wav,
                threshold_a,
                threshold_d,
                log_callback
            )

            # 3. Print segments
            self._print_segments(silent_segments, log_callback)

            # 4. Construct FFmpeg command
            keep_segments = self._manage_commands(silent_segments, path_obj, log_callback)

            # 5. Trimming command
            cmd = self._make_commands(keep_segments, path_obj, output_video)
            log_callback("5. Final trimming command is \n" + " ".join(cmd))

            # 6. Execute Trimming
            log_callback(f"6: Executing FFmpeg trim... (Saving to {output_video.name})")
            log_callback("This process can be long, please do not quit.")

            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            log_callback("Processing Complete.")

        finally:
            # Cleanup temp file
            if os.path.exists(temp_wav):
                os.remove(temp_wav)


    def _extract_audio(self, path_obj:Path, temp_wav:Path, log_callback) -> None:
        # 1. Extract Audio
        log_callback("1. Extracting audio via FFmpeg...")
        self.extract_audio(str(path_obj), str(temp_wav))


    def _analysis_audio(self,
                        temp_wav: Path,
                        threshold_a: float,
                        threshold_d: float,
                        log_callback)\
            -> list[tuple[float, float]] :
        # 2. Analyze Audio
        log_callback("2. Analyzing audio waveform with Scipy...")
        silent_segments = self.analyzer.find_silent_segments(
            str(temp_wav),
            threshold_a,
            threshold_d)
        return silent_segments

    @staticmethod
    def _print_segments(silent_segments, log_callback):
        log_callback(f"3. Found {len(silent_segments)} silent segments.")
        for i, (start, end) in enumerate(silent_segments):
            log_callback(f"  - Silence {i + 1}: {start:.2f}s to {end:.2f}s")

        if not silent_segments:
            log_callback("No silence detected matching criteria. No trimming performed.")


    def _manage_commands(self,
                         silent_segments: list[tuple[float, float]],
                         path_obj: Path,
                         log_callback
                         )-> list[tuple[float, float]]:
        log_callback("Step 4: Constructing trim command...")

        total_duration = self._get_video_duration(str(path_obj), log_callback=log_callback)
        keep_segments: list[tuple[float, float]] = []
        current_time = 0.0

        for s_start, s_end in silent_segments:
            if s_start > current_time:
                keep_segments.append((current_time, s_start))
            current_time = max(current_time, s_end)

        # Add the final chunk after the last silence if applicable
        if current_time < total_duration:
            keep_segments.append((current_time, total_duration))
        return keep_segments

    def _make_commands(
            self,
            keep_segments: list[tuple[float, float]],
            path_obj: Path,
            output_video: Path
    )-> list[str]:

        """
        Make the final cmd for removing the silence based on
        keep_segments (segments to keep), target video path_obj,
        to be output in output_video file path
        (FFMPEG dark magic)

        :param keep_segments: segments to keep (in seconds) list of tuples [start/end]
        :param path_obj: Path to the target video (Path object)
        :param output_video: Path to the output video (Path object)
        :return: command (list of strings)
        """

        # Build complex filter
        # Format:
        # [0:v]trim=start=0:end=10,setpts=PTS-STARTPTS[v0];...
        # [0:a]atrim=start=0:end=10,asetpts=PTS-STARTPTS[a0];...
        filter_parts = []
        outputs = []

        for i, (start, end) in enumerate(keep_segments):
            # Video trim
            filter_parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}]")
            # Audio trim
            filter_parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]")
            outputs.append(f"[v{i}][a{i}]")

        # Concat part
        concat_input = "".join(outputs)
        filter_parts.append(f"{concat_input}concat=n={len(keep_segments)}:v=1:a=1[outv][outa]")

        filter_complex = ";".join(filter_parts)

        # 5. Trimming command
        cmd = [
            self.ffmpeg_cmd, "-y",
            "-i", str(path_obj),  # path
            "-filter_complex", filter_complex,  # filter with concat
            "-map", "[outv]", "-map", "[outa]",  # out file
            str(output_video)  # out file path
        ]
        return cmd

class MainWindow: # pylint: disable=R0902
    """
    View class responsible for the Tkinter Interface.
    It knows nothing about FFmpeg or Scipy.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Video Silence Remover")
        self.root.geometry("600x500")

        # Variables
        self.file_path_var = tk.StringVar()

        default_silence_duration_threshold = 0.4   # Default 0.4 second
        default_amplitude_threshold = 0.05 # Default 5% amplitude

        self.threshold_d_var = tk.DoubleVar(value=default_silence_duration_threshold)
        self.threshold_a_var = tk.DoubleVar(value=default_amplitude_threshold)

        self._create_widgets()

    def _create_widgets(self):
        # File Selection Frame
        frame_file = tk.LabelFrame(self.root, text="Input File", padx=10, pady=10)
        frame_file.pack(fill="x", padx=10, pady=5)

        tk.Entry(frame_file, textvariable=self.file_path_var, width=50).pack(side="left", padx=5)
        tk.Button(frame_file, text="Browse", command=self.on_browse).pack(side="left")

        # Settings Frame
        frame_settings = tk.LabelFrame(self.root, text="Settings", padx=10, pady=10)
        frame_settings.pack(fill="x", padx=10, pady=5)

        # Threshold Duration (D)
        (tk.Label(frame_settings, text="Threshold Duration (D) [sec]:")
         .grid(row=0, column=0, sticky="w"))
        (tk.Entry(frame_settings, textvariable=self.threshold_d_var)
         .grid(row=0, column=1, sticky="w", padx=5))

        # Threshold Amplitude (A)
        # Create a Label to show the percentage
        lbl_value = (
            tk.Label(frame_settings, text=f"Amplitude threshold: {self.threshold_a_var.get():.0%}"))
        lbl_value.grid()

        # Define a function to update the label
        def update_lbl(value):
            # value comes in as a string from the scale event
            float_val = float(value)
            # Format as percentage (e.g., 0.5 -> "50%")
            lbl_value.config(text=f"Amplitude threshold: {float_val:.0%}")

        # Create the Scale
        # set showvalue=0 to hide the default "0.55" display
        tk.Scale(frame_settings,
                 variable=self.threshold_a_var,
                 from_=1, to=0,
                 resolution=0.01,
                 orient="vertical",
                 label="Amplitude Threshold",
                 showvalue=False,  # <--- Hides the default number
                 command=update_lbl  # Triggers label update
                 ).grid(row=1, column=1)

        # Run Button
        self.btn_run = tk.Button(self.root, text="Run Calculation",
                                 command=self.on_run, bg="#DDDDDD", height=2)
        self.btn_run.pack(fill="x", padx=10, pady=10)

        # Run Button
        self.btn_show = tk.Button(self.root, text="Preview",
                                  command=self.on_show, bg="#DDDDDD", height=2)
        self.btn_show.pack(fill="x", padx=10, pady=10)

        # Log Console
        tk.Label(self.root, text="Console Output:").pack(anchor="w", padx=10)
        self.console = scrolledtext.ScrolledText(self.root, height=12, state='disabled')
        self.console.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Event callbacks (to be assigned by Controller)
        self.browse_callback = None
        self.run_callback = None

    def on_browse(self):
        """
        Function when clicking on the browse file button
        :return:
        """
        if self.browse_callback:
            self.browse_callback()

    def on_run(self):
        """
        Function when clicking on run
        :return:
        """
        if self.run_callback:
            self.run_callback()

    def on_show(self):
        """
        Function when clicking on preview
        :return:
        """
        dict_inputs = self.get_inputs()
        AudioAnalyzer().graph_plot(
            dict_inputs["filepath"],
            dict_inputs["threshold_a"],
            dict_inputs["threshold_d"]
        )

    # Public methods for Controller to update View
    def set_file_path(self, path):
        """
        Method to set the file path
        :param path:
        :return:
        """
        self.file_path_var.set(path)

    def get_inputs(self):
        """
        Method to get the content of the config dict
        :return:
        """
        return {
            'filepath': self.file_path_var.get(),
            'threshold_d': self.threshold_d_var.get(),
            'threshold_a': self.threshold_a_var.get()
        }

    def log(self, message): # pylint: disable=C0116
        self.console.config(state='normal')
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)
        self.console.config(state='disabled')

    @staticmethod
    def show_error(title, message) -> None: # pylint: disable=C0116
        messagebox.showerror(title, message)

    @staticmethod
    def show_info(title, message) -> None: # pylint: disable=C0116
        messagebox.showinfo(title, message)

    def toggle_run_button(self, state: bool) -> None:
        """
        Toggle the button Run to a certain state in argument
        (Literal["normal","active","disabled"])
        :param state: (Literal["normal","active","disabled"])
        :return:
        """
        self.btn_run.config(state=state)


class AppController:
    """
    Controller class connecting the View and the Model.
    Handles threading to prevent UI freeze.
    """

    def __init__(self, root: tk.Tk):
        self.view = MainWindow(root)
        self.model = VideoProcessor()

        # Bind View events
        self.view.browse_callback = self.handle_browse
        self.view.run_callback = self.handle_run

    def handle_browse(self) -> None:
        """
        Handles the browsing (filedialog) and select a video file
        :return:
        """
        filetypes = (
            ("Video files", "*.mp4 *.avi *.mkv"),
            ("All files", "*.*")
        )
        filename = filedialog.askopenfilename(title="Select Video File", filetypes=filetypes)
        if filename:
            self.view.set_file_path(filename)

    def handle_run(self) -> None:
        """
        Triggers the processing using threading
        :return:
        """
        inputs = self.view.get_inputs()

        # Validation
        filepath = inputs['filepath']
        if not filepath:
            self.view.show_error("Input Error", "Please select a video file.")
            return

        try:
            threshold_d = float(inputs['threshold_d'])
            threshold_a = float(inputs['threshold_a'])
        except ValueError:
            self.view.show_error("Input Error", "Thresholds must be numbers.")
            return

        # Disable button during processing
        self.view.toggle_run_button(tk.DISABLED)
        self.view.log("--- Starting Process ---")

        # Threading: Run heavy processing in background
        process_thread = threading.Thread(
            target=self._run_process_thread,
            args=(filepath, threshold_a, threshold_d),
            daemon=True
        )
        process_thread.start()

    def _run_process_thread(self, filepath: str, threshold_a: float, threshold_d: float):

        try:
            # Delegate to Facade Model
            self.model.process_video(
                filepath,
                threshold_a,
                threshold_d,
                log_callback=self._update_log_safe
            )
            # Success Feedback
            self._update_log_safe("Done!")
            self.view.root.after(0, lambda: self.view.show_info("Success", "Video file created."))
        except Exception as e: # pylint: disable=W0718
            err_msg = str(e)
            self._update_log_safe(f"Error: {err_msg}")
            self.view.root.after(0, lambda: self.view.show_error("Execution Error", err_msg))
        finally:
            self.view.root.after(0, lambda: self.view.toggle_run_button(tk.NORMAL))

    def _update_log_safe(self, message):
        """Thread-safe update of UI log."""
        self.view.root.after(0, lambda: self.view.log(message))

from setuptools import setup

setup(
    name='silence_remover_ad',
    version='1.0.0',
    author='Alexandre Delaisement',
    author_email='',
    description='A module to remove silence part with a user-friendly UI',
    long_description='Raw video footage can be a hassle to process; and in particular long periods of silences can be superfluous.\nThe goal of this Python module is to provide a tool to remove video footage from the "silent part".\n This can be used by humans to increase density of information of their video, or to train AI models only on unsilenced part.\n It revolves heavily around "ffmpeg".',
    long_description_content_type='text/markdown',
    url='https://github.com/AlexandreDela/silence_remover_ad',
    packages=['silence_remover_ad'],
      package_dir={'silence_remover_ad': 'src/silence_remover_ad'},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        'Topic :: Multimedia',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Multimedia :: Sound/Audio :: Conversion',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
        'Topic :: Multimedia :: Video',
        'Topic :: Multimedia :: Video :: Conversion',
        'Topic :: Utilities',
        'Typing :: Typed',
    ],
    keywords=['video','audio','silence','silent','remover', 'remove','ffmpeg'],
    python_requires='>=3.8',
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    project_urls={
        'Source': 'https://github.com/AlexandreDela/Silence_remover_ad',
    },
)
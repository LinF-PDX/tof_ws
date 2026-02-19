from setuptools import setup

package_name = 'tof_ground_seg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@todo.com',
    description='ToF ground segmentation into grid cells with per-cell plane normals.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'grid_ground = tof_ground_seg.grid_ground_node:main',
            'timing_hist = tof_ground_seg.timing_histogram_from_logs:main',
            'timing_hist_txt = tof_ground_seg.plot_timing_txt:main',
        ],
    },
)

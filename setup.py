from setuptools import setup, find_packages

setup(
    name='AV-piper',                # 发行包名
    version='0.1.0',
    packages=find_packages(),       # 自动扫描当前目录下所有带 __init__.py 的文件夹（如 env, agent 等）
    include_package_data=True,      # 允许打包非 Python 文件（如你的 XML 模型文件）
    package_data={
        'env': [
            'assets/*',
            'assets/meshes/*',
            'assets/meshes/piperx/*',
            'assets/meshes/cameras/*',
            'assets/old/*',
        ],
    },
    entry_points={
        'mjlab.tasks': [
            'av_piper=env.mjlab',
        ],
    },
    description='AV-piper Local Package',
)

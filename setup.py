from setuptools import find_namespace_packages, find_packages, setup


project_packages = find_packages()
vendored_lerobot_packages = find_namespace_packages(
    include=["lerobot", "lerobot.*"],
)

setup(
    name='AV-piper',                # 发行包名
    version='0.1.0',
    # LeRobot 0.1.0 的 common 等目录使用隐式 namespace
    # package；仅使用 find_packages() 会在构建 wheel 时漏掉这些模块。
    packages=sorted(set(project_packages + vendored_lerobot_packages)),
    include_package_data=True,      # 允许打包非 Python 文件（如你的 XML 模型文件）
    package_data={
        'env': [
            'assets/*',
            'assets/meshes/*',
            'assets/meshes/piperx/*',
            'assets/meshes/cameras/*',
            'assets/old/*',
        ],
        'lerobot': ['LICENSE'],
    },
    description='AV-piper Local Package',
)

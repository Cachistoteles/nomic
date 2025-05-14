def _check_aws_dependencies():
    try:
        pass
    except ImportError as e:
        missing_package = str(e).split("No module named ")[-1].strip("'")
        raise ImportError(
            f"The '{missing_package}' package is required for this feature. "
            "Please install it by running 'pip install nomic[aws]'."
        ) from e


_check_aws_dependencies()

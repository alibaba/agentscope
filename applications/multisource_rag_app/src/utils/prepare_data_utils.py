# -*- coding: utf-8 -*-
"""
Prepare tutorial data of AgentScope
"""
import os


def prepare_docstring_txt(repo_path: str, text_dir: str) -> None:
    """
    If repo_path and text_dir are provided, and text_dir is empty,
    it prepares the docstring in html, and save it.

    Args:
        repo_path (`str`):
            The path of the repo
        text_dir (`str`):
            The path of the text dir
    Returns:
        None
    """
    print(f"AS repo path: {repo_path}, text_dir: {text_dir}")
    if (
        len(repo_path) > 0
        and len(text_dir) > 0
        and os.path.exists(repo_path)
        and not os.path.exists(text_dir)
    ):
        os.system(
            f"sphinx-apidoc -f -o {repo_path}/docs/tutorial/"
            "en/source/build_api"
            f" {repo_path}/src/agentscope "
            f"-t {repo_path}/docs/tutorial/en/source/_templates",
        )
        os.system(
            f"sphinx-build -M text  {repo_path}/docs/tutorial/en/source "
            f"{text_dir} -W --keep-going",
        )

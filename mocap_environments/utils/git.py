import json
import pathlib

import git

Path = pathlib.Path


def save_git_info(repo_path: Path, info_dir_path: Path):
    repo = git.Repo(repo_path)
    repos = {"main": repo} | {
        submodule.name: git.Repo(submodule.abspath) for submodule in repo.submodules
    }
    info_dir_path.mkdir(parents=True, exist_ok=True)
    for repo_name, repo in repos.items():
        patch_file_path = (info_dir_path / repo_name).with_suffix(".patch")
        diff = repo.git.diff("HEAD")
        with patch_file_path.open("wt") as f:
            f.write(diff)

    git_revs = {k: get_git_rev(v) for k, v in repos.items()}
    with (info_dir_path / "revs.json").open("wt") as f:
        json.dump(git_revs, f, indent=2, sort_keys=True)


def get_git_rev(repo: git.Repo) -> str:
    if repo.head.is_detached:
        git_rev = repo.head.object.name_rev
    else:
        git_rev = repo.active_branch.commit.name_rev

    return git_rev

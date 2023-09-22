from pathlib import Path

from configs.settings import BASE_DIR


def get_log_folder_name(args):
    folder_name = "-".join(
        [
            args.data_type,
            args.attack_name,
            args.dataset,
            args.model,
            "pratio=%s" % args.pratio,
        ]
    )
    default_path = BASE_DIR / "logs" / folder_name
    if not Path.exists(default_path):
        Path.mkdir(default_path)
    return default_path


def get_log_folder_name_by_info(data_type, attack_name, dataset, model, pratio):
    folder_name = "-".join(
        [data_type, attack_name, dataset, model, "pratio=%s" % pratio]
    )
    default_path = BASE_DIR / "logs" / folder_name
    if not Path.exists(default_path):
        Path.mkdir(default_path)
    return default_path


def get_defense_log_name():
    pass

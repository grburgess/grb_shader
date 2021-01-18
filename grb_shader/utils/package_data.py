from pathlib import Path
import pkg_resources

def get_path_of_data_file(data_file) -> Path:
    file_path = pkg_resources.resource_filename("grb_shader", "data/%s" % data_file)

    return Path(file_path)


# def get_path_to_template_analysis():

#     return get_path_of_data_file("template.md")


# def get_path_to_database() -> Path:

#     p: Path = Path(gbm_kitty_config["database"])

#     if not p.exists():

#         p.mkdir(parents=True)

#     return p

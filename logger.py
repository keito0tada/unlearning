import datetime
import time
from logging import getLogger, config
import yaml
import jinja2

with open("logging.yaml", "r", encoding="utf-8") as f:
    config.dictConfig(
        yaml.safe_load(
            jinja2.Template(f.read()).render(
                {"filepath": f"log/{datetime.date.today()}.log"}
            )
        )
    )
logger_regular = getLogger("regular")
logger_overwrite = getLogger("overwrite")

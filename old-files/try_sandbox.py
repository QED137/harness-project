from pathlib import Path

import pandas as pd

from tsagent.sandbox import DockerSandbox, SandboxConfig

Path("data").mkdir(exist_ok=True)
pd.DataFrame({"temp": [10.5, 12.0, 14.2]}).to_parquet("data/weather.parquet")

sb = DockerSandbox(SandboxConfig(data_dir=Path("data"), preload="/data/weather.parquet"))

for code in ["result = df['temp'].mean()", "import os", "while True: pass"]:
    r = sb.run(code)
    print(f"{code!r:30} -> {r.status.value:16} {r.result or r.error}")
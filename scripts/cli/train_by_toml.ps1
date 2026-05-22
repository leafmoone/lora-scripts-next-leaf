# LoRA train script by @Akegarasu
# DEPRECATED: Anima / ?????????? WebUI (run_gui.bat)??????????? SD1.5/SDXL ??? CLI ?¦Ï???

$config_file = "./config/default.toml"		 # config file | ??? toml ?????????????
$sample_prompts = "./config/sample_prompts.txt"		 # prompt file for sample | ???? prompts ???, ?????????¨°???????

$sdxl = 0        # train sdxl LoRA | ??? SDXL LoRA
$multi_gpu = 0		 # multi gpu | ???????? ?¨°?????????????? >= 2 ???

# ============= DO NOT MODIFY CONTENTS BELOW | ????????¡¤????? =====================

# Activate python venv
.\venv\Scripts\activate

$Env:HF_HOME = "huggingface"
$Env:PYTHONUTF8 = 1

$ext_args = [System.Collections.ArrayList]::new()
$launch_args = [System.Collections.ArrayList]::new()

if ($multi_gpu) {
  [void]$launch_args.Add("--multi_gpu")
  [void]$launch_args.Add("--num_processes=2")
}

$trainer_file = if ($sdxl) { "./vendor/sd-scripts/sdxl_train_network.py" } else { "./vendor/sd-scripts/train_network.py" }
python -m accelerate.commands.launch $launch_args --num_cpu_threads_per_process=8 $trainer_file `
  --config_file=$config_file `
  --sample_prompts=$sample_prompts `
  $ext_args

Write-Output "Train finished"
Read-Host | Out-Null

import os
import pathlib

# Set the new HuggingFace cache directory
new_cache_dir = "/media/monsterdrive/models"

# Create the directory if it doesn't exist
pathlib.Path(new_cache_dir).mkdir(parents=True, exist_ok=True)

# Set the environment variable for the current Python process
os.environ["HF_HOME"] = new_cache_dir

# Create a shell script that can be sourced
shell_script_path = "set_hf_home.sh"
with open(shell_script_path, "w") as f:
    f.write(f'export HF_HOME="{new_cache_dir}"\n')

print(f"HuggingFace cache directory set to: {new_cache_dir}")
print("\nTo apply this in your current terminal session, run:")
print(f"source {shell_script_path}")
print("\nTo set this permanently, add the following to your ~/.bashrc:")
print(f'export HF_HOME="{new_cache_dir}"')
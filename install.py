import launch

if not launch.is_installed("bitsandbytes"):
    launch.run_pip("install bitsandbytes", "requirements for Creative Prompt Enhancer")

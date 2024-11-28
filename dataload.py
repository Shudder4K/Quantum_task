import kagglehub

# Download latest version
path = kagglehub.dataset_download("isaienkov/deforestation-in-ukraine")

print("Path to dataset files:", path)
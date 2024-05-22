import zipfile
with zipfile.ZipFile("models - checkpoint 96.zip","r") as zip_ref:
    zip_ref.extractall("models")

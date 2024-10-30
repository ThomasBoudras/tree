import pkg_resources

# Chemin vers ton fichier requirements d'origine sans versions
input_file = "/home/projects/ku_00196/people/thobou/tree/requirements.txt"
output_file = input_file

# Lire le fichier original sans versions
with open(input_file, "r") as f:
    packages = f.read().splitlines()

# Créer un nouveau fichier avec les versions ajoutées
with open(output_file, "w") as f:
    for package in packages:
        if package.strip() and not package.startswith("#"):  # Ignorer les lignes vides et les commentaires
            try:
                # Récupérer la version installée
                version = pkg_resources.get_distribution(package).version
                f.write(f"{package}=={version}\n")
            except pkg_resources.DistributionNotFound:
                f.write(f"{package} # Version not found\n")
                print(f"Warning: {package} n'est pas installé dans l'environnement actuel.")
        else:
            f.write(package + "\n")  # Conserver les commentaires et lignes vides

print(f"Le fichier {output_file} a été généré avec les versions.")

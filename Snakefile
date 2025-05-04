import pandas as pd

# Configuration
configfile: "config.yaml"

# Charger les paramètres du CSV
df = pd.read_csv("params.csv", index_col="name")
df.columns = df.columns.str.strip()  # Supprime les espaces autour des noms de colonnes
params = df.T.to_dict()

# Chemin vers yapt
yapt_path = lambda wildcards, config: config["yapt_path"]

rule all:
    input:
        "results/stats_summary.csv"

rule run_yapt:
    output:
        mapfile="maps/{name}.txt"
    params:
        yapt=config["yapt_path"],
        tmpdir=lambda wildcards: f"tmp/{wildcards.name}",
        args=lambda wildcards: (
            f"source={config['function_path']}/{params[wildcards.name]['source']} "
            f"spp={params[wildcards.name]['spp']}"
        )
    shell:
        r"""
        set -x
        mkdir -p {params.tmpdir}
        mkdir -p {config[output_image_path]}
        {params.yapt}/yapt dir={params.tmpdir} {params.args}

        # Trouver le seul fichier .exr produit
        exr_file=$(find {params.tmpdir} -maxdepth 1 -name "*.exr" | head -n 1)

        if [ -z "$exr_file" ]; then
            echo "Aucun fichier .exr généré !" >&2
            exit 1
        fi

        # Extraire le nom du fichier seulement
        filename=$(basename "$exr_file")

        # Enregistrer le nom dans mapfile
        echo "$filename" > {output.mapfile}

        # Déplacer vers le dossier final
        mv "$exr_file" {config[output_image_path]}/"$filename"
        """


rule extract_stats:
    input:
        mapfile="maps/{name}.txt",        
    output:
        "stats/{name}.txt"
    run:
        import sys
        with open(input.mapfile) as f:
            image_name = f.read().strip()
        print(f"→ Traitement de {image_name}", file=sys.stderr)
        shell(f"""      
            oiiotool --stats {config['output_image_path']}/{image_name} | grep 'Avg' > {output[0]}
        """)

rule aggregate_stats:
    input:
        expand("stats/{name}.txt", name=params.keys())
    output:
        "results/stats_summary.csv"
    run:
        import re
        with open(output[0], "w") as out:
            out.write("name,mean\n")
            for infile in input:
                name = infile.split("/")[-1].replace(".txt", "")
                with open(infile) as f:
                    content = f.read()
                    # Recherche le premier nombre après "Avg:"
                    match = re.search(r"Avg:\s+([0-9.eE+-]+)", content)
                    mean = match.group(1) if match else "NaN"
                    out.write(f"{name},{mean}\n")

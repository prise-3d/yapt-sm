import pandas as pd

import os
FUNCTIONS = [
    "unit_disk", "f1", "f2", "disturbed_disk", "diamond",
    "periodical1", "periodical2", "periodical3", "periodical4",
    "gaussian", "ellipse", "fractal1", "fractal2", "hard1", "hard2"
]

# Configuration
configfile: "config.yaml"

# Load paramètre CSV
df = pd.read_csv(config["params_file"], index_col="name")
df.columns = df.columns.str.strip()  # Nettoyage des espaces dans les noms de colonnes
params = df.T.to_dict()

# Extraire les fonctions depuis la colonne 'source' (en enlevant les extensions)
FUNCTIONS = sorted(set(os.path.splitext(os.path.basename(src))[0] for src in df["source"]))
print("Fonctions trouvées :", FUNCTIONS)


rule all:
    input:
        "results/stats_summary.csv",
        "results/plots_by_function/plot_unit_disk.png",
        "results/plots_by_function/plot_f1.png",
        "results/all_plots.pdf",
        "results/stats_summary.csv",
        "results/tables.md",
        expand("results/plots_by_function/plot_{function}.png", function=FUNCTIONS)
        

# Rule to run yapt and process .exr files
rule run_yapt:
    output:
        mapfile="maps/{name}.txt",
        timefile="results/times/{name}.txt"
    params:
        yapt=config["yapt_path"],
        tmpdir=lambda wildcards: f"tmp/{wildcards.name}",
        args=lambda wildcards: (
            f"source={config['function_path']}/{params[wildcards.name]['source']} "
            f"spp={params[wildcards.name]['spp']} "
            f"sampler={params[wildcards.name]['sampler']} "
            f"aggregator={params[wildcards.name]['aggregator']} "                    
        )
    shell:
        r"""
        set -x
        mkdir -p {params.tmpdir}
        mkdir -p {config[output_image_path]}
        mkdir -p results/times

        # Measure wall-clock time in seconds and store it
        /usr/bin/time -f "%e" -o {output.timefile} {params.yapt}/yapt dir={params.tmpdir} {params.args}

        # Find the single .exr file produced
        exr_file=$(find {params.tmpdir} -maxdepth 1 -name "*.exr" | head -n 1)

        if [ -z "$exr_file" ]; then
            echo "No .exr file created!" >&2
            exit 1
        fi

        # Extract only the file name
        filename=$(basename "$exr_file")

        # Save the name in mapfile
        echo "$filename" > {output.mapfile}

        # Move to the final folder
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
        print(f"→ Processing {image_name}", file=sys.stderr)
        shell(f"""      
            oiiotool --stats {config['output_image_path']}/{image_name} > {output}
        """)

rule aggregate_stats:
    input:
        statfiles=expand("stats/{name}.txt", name=params.keys()),
        timefiles=expand("results/times/{name}.txt", name=params.keys())
    output:
        "results/stats_summary.csv"
    params:
        all_params=params
    run:
        import re
        import pandas as pd
        import os

        rows = []
        print(f"→ Aggregating stats from {len(input.statfiles)} files")
        # display the list of files
        print("→ Stat files:", input.statfiles)
        print("→ Time files:", input.timefiles)
        
        # Itérer seulement sur les fichiers de statistiques
        for stat_file in input.statfiles:
            name = os.path.basename(stat_file).replace(".txt", "")
            print(f"→ Aggregating {name}")
            all_params = params["all_params"]
            if name not in all_params:
                print("Nom de fichier inattendu :", name)
                print("Fichier source :", stat_file)
                print("Clés disponibles :", list(params.keys()))
                raise ValueError(f"Paramètres introuvables pour {name}")
            param = all_params[name]
            
            # Lecture des stats .exr
            with open(stat_file) as f:
                content = f.read()

            stats = {}
            for stat in ["Min", "Max", "Avg", "StdDev"]:
                match = re.search(rf"{stat}:\s+([0-9.eE+-]+)", content)
                stats[stat.lower()] = float(match.group(1)) if match else float("nan")

            # Lecture du temps d'exécution
            time_file = f"results/times/{name}.txt"
            try:
                with open(time_file) as tf:
                    stats["time"] = float(tf.read().strip())
            except:
                stats["time"] = float("nan")

            stats["name"] = name
            stats["spp"] = param["spp"]
            stats["source"] = os.path.splitext(os.path.basename(param["source"]))[0]
            stats["aggregator"] = param["aggregator"]
            stats["sampler"] = param["sampler"]
            rows.append(stats)            

        df = pd.DataFrame(rows)
        df = df[["name", "source", "spp", "min", "max", "avg", "stddev", "time", "aggregator", "sampler"]]
        df.to_csv(output[0], index=False)


rule plot_stats:
    input:
        stats="results/stats_summary.csv",
        params=config.get("params_file", "params.csv")
    output:
        "results/plots_by_function/plot_{function}.png"
    run:
        import matplotlib
        matplotlib.use("Agg")

        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import os
        import math

        function_name = wildcards.function

        # Load and merge
        df_stats = pd.read_csv(input.stats)
        print(f"→ Stats CSV shape: {df_stats.shape}")
        print(f"→ Stats CSV columns: {list(df_stats.columns)}")
        print(f"→ Premier échantillon de stats:")
        print(df_stats.head())
        
        df_params = pd.read_csv(input.params)
        df_params.columns = df_params.columns.str.strip()
        extra_cols = [col for col in df_params.columns if col != "name" and col not in df_stats.columns]
        df = pd.merge(df_stats, df_params[["name"] + extra_cols], on="name")
        df.columns = df.columns.str.strip()

        print(f"→ Recherche de données pour la fonction '{function_name}'")
        print(f"→ DataFrame final shape: {df.shape}")
        print(f"→ Colonnes du DataFrame: {list(df.columns)}")
        print(f"→ Valeurs uniques dans 'source': {sorted(df['source'].unique())}")
        print(f"→ Valeurs uniques dans 'spp': {sorted(df['spp'].unique())}")
        
        # Filter by function name - utiliser une correspondance exacte ET partielle
        print(f"→ Tentative de correspondance exacte avec '{function_name}'")
        df_filtered = df[df["source"] == function_name]
        print(f"→ Correspondance exacte: {len(df_filtered)} lignes")
        
        if df_filtered.empty:
            print(f"→ Tentative de correspondance partielle avec '{function_name}'")
            df_filtered = df[df["source"].str.contains(function_name, case=False, na=False)]
            print(f"→ Correspondance partielle: {len(df_filtered)} lignes")
        
        print(f"→ Données avant filtrage spp: {len(df_filtered)} lignes")
        if not df_filtered.empty:
            print(f"→ Valeurs spp pour cette fonction: {sorted(df_filtered['spp'].unique())}")
        
        # Filtrer par spp
        df_filtered = df_filtered[df_filtered["spp"] > 10].sort_values("spp")
        print(f"→ Données après filtrage spp > 10: {len(df_filtered)} lignes")

        if df_filtered.empty:
            print(f"→ Aucune donnée trouvée pour '{function_name}', création d'un graphique vide")
            
            # Créer un graphique vide avec un message d'information
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Aucune donnée disponible\npour la fonction '{function_name}'", 
                   ha='center', va='center', fontsize=16, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"Fonction: {function_name}", fontsize=14)
            ax.axis('off')
            
            os.makedirs(os.path.dirname(output[0]), exist_ok=True)
            fig.savefig(output[0], dpi=150, bbox_inches='tight')
            plt.close(fig)
            return

        df = df_filtered
        print(f"→ {len(df)} lignes de données trouvées pour '{function_name}'")

        # Séparer les agrégateurs
        mc_df = df[df["aggregator"] == "mc"]
        other_aggs = sorted(set(df["aggregator"]) - {"mc"})
        
        if not other_aggs:
            print(f"→ Seules les données 'mc' trouvées pour '{function_name}'")
            # Si seulement mc, créer un graphique simple
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Graphique des moyennes
            for sampler, sub_group in mc_df.groupby("sampler"):
                sub_group = sub_group.sort_values("spp")
                sns.lineplot(ax=ax1, x="spp", y="avg", data=sub_group, marker="o", label=f"{sampler}")
                ax1.fill_between(sub_group["spp"],
                                sub_group["avg"] - sub_group["stddev"],
                                sub_group["avg"] + sub_group["stddev"],
                                alpha=0.3)
            
            ax1.set_title(f"Mean - Aggregator: mc")
            ax1.set_xscale("log")
            ax1.set_xlabel("Samples per pixel (SPP)")
            ax1.set_ylabel("Mean")
            ax1.grid(True)
            ax1.legend()
            
            # Graphique des temps
            for sampler, sub_group in mc_df.groupby("sampler"):
                sub_group = sub_group.sort_values("spp")
                sns.lineplot(ax=ax2, x="spp", y="time", data=sub_group, marker="o", label=f"{sampler}")
            
            ax2.set_title(f"Execution Time - Aggregator: mc")
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_xlabel("Samples per pixel (SPP)")
            ax2.set_ylabel("Time (seconds)")
            ax2.grid(True)
            ax2.legend()
            
            fig.suptitle(f"Function: {function_name} (mc only)", fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            
        else:
            n = len(other_aggs)
            cols = 2
            rows = math.ceil(n / cols)

            # Créer une figure avec 2 rangées : une pour les moyennes, une pour les temps
            fig = plt.figure(figsize=(6 * cols, 8 * rows))

            # Première partie : graphiques des moyennes
            for i, aggregator in enumerate(other_aggs):
                ax = plt.subplot(2 * rows, cols, i + 1)

                # Données pour cet agrégateur
                agg_df = df[df["aggregator"] == aggregator]

                # Courbes pour agrégateur courant
                for sampler, sub_group in agg_df.groupby("sampler"):
                    sub_group = sub_group.sort_values("spp")
                    sns.lineplot(ax=ax, x="spp", y="avg", data=sub_group, marker="o", label=f"{sampler}")

                    ax.fill_between(sub_group["spp"],
                                    sub_group["avg"] - sub_group["stddev"],
                                    sub_group["avg"] + sub_group["stddev"],
                                    alpha=0.3)

                # Courbes pour 'mc' en référence (si disponible)
                if not mc_df.empty:
                    for sampler, sub_group in mc_df.groupby("sampler"):
                        sub_group = sub_group.sort_values("spp")
                        sns.lineplot(ax=ax, x="spp", y="avg", data=sub_group, linestyle="--", marker="x", label=f"mc/{sampler}")

                        ax.fill_between(sub_group["spp"],
                                        sub_group["avg"] - sub_group["stddev"],
                                        sub_group["avg"] + sub_group["stddev"],
                                        alpha=0.15)

                ax.set_title(f"Mean - Aggregator: {aggregator}")
                ax.set_xscale("log")
                ax.set_xlabel("Samples per pixel (SPP)")
                ax.set_ylabel("Mean")
                ax.grid(True)
                ax.legend()

            # Deuxième partie : graphiques des temps d'exécution
            for i, aggregator in enumerate(other_aggs):
                ax = plt.subplot(2 * rows, cols, rows * cols + i + 1)

                # Données pour cet agrégateur
                agg_df = df[df["aggregator"] == aggregator]

                # Courbes pour agrégateur courant
                for sampler, sub_group in agg_df.groupby("sampler"):
                    sub_group = sub_group.sort_values("spp")
                    sns.lineplot(ax=ax, x="spp", y="time", data=sub_group, marker="o", label=f"{sampler}")

                # Courbes pour 'mc' en référence (si disponible)
                if not mc_df.empty:
                    for sampler, sub_group in mc_df.groupby("sampler"):
                        sub_group = sub_group.sort_values("spp")
                        sns.lineplot(ax=ax, x="spp", y="time", data=sub_group, linestyle="--", marker="x", label=f"mc/{sampler}")

                ax.set_title(f"Execution Time - Aggregator: {aggregator}")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("Samples per pixel (SPP)")
                ax.set_ylabel("Time (seconds)")
                ax.grid(True)
                ax.legend()

            fig.suptitle(f"Mean & Execution Time vs SPP by Sampler — Function: {function_name}\n(mc shown on all)", fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        fig.savefig(output[0], dpi=150, bbox_inches='tight')
        plt.close(fig)


import glob
import matplotlib.pyplot as plt  # Added import for plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image        

rule combine_plots:
    input:
        expand("results/plots_by_function/plot_{function}.png", function=FUNCTIONS)
    output:
        "results/all_plots.pdf"
    run:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        from PIL import Image
        import os

        os.makedirs(os.path.dirname(output[0]), exist_ok=True)

        with PdfPages(output[0]) as pdf:
            for img_path in input:
                image = Image.open(img_path)
                fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100))
                ax.imshow(image)
                ax.axis("off")
                pdf.savefig(fig)
                plt.close(fig)        

rule generate_markdown_tables:
    input:
        stats="results/stats_summary.csv",
        params=config.get("params_file", "params.csv")
    output:
        "results/tables.md"
    run:
        import pandas as pd
        import os
        
        # Charger les données
        df_stats = pd.read_csv(input.stats)
        df_params = pd.read_csv(input.params)
        df_params.columns = df_params.columns.str.strip()
        
        # Fusionner les données
        extra_cols = [col for col in df_params.columns if col != "name" and col not in df_stats.columns]
        df = pd.merge(df_stats, df_params[["name"] + extra_cols], on="name")
        df.columns = df.columns.str.strip()
        
        # Début du document Markdown
        md_content = [
            "# Résultats des intégrateurs par fonction",
            "",
            f"Rapport généré automatiquement à partir de {len(df)} configurations.",
            ""
        ]
        
        # Pour chaque fonction
        functions = sorted(df["source"].unique())
        
        for function_name in functions:
            df_function = df[df["source"] == function_name]
            
            if df_function.empty:
                continue
                
            md_content.extend([
                f"## Fonction: {function_name}",
                ""
            ])
            
            # Grouper par agrégateur
            aggregators = sorted(df_function["aggregator"].unique())
            
            for aggregator in aggregators:
                df_agg = df_function[df_function["aggregator"] == aggregator]
                
                if df_agg.empty:
                    continue
                
                md_content.extend([
                    f"### Agrégateur: {aggregator}",
                    ""
                ])
                
                # En-tête du tableau
                md_content.extend([
                    "| Sampler | SPP | Moyenne | Écart-type | Temps (s) |",
                    "|---------|-----|---------|------------|-----------|"
                ])
                
                # Trier par sampler puis par spp
                df_agg_sorted = df_agg.sort_values(["sampler", "spp"])
                
                for _, row in df_agg_sorted.iterrows():
                    sampler = row["sampler"]
                    spp = int(row["spp"])
                    avg = f"{row['avg']:.6f}"
                    stddev = f"{row['stddev']:.6f}"
                    time_val = f"{row['time']:.3f}"
                    
                    md_content.append(f"| {sampler} | {spp} | {avg} | {stddev} | {time_val} |")
                
                md_content.append("")  # Ligne vide après chaque tableau
        
        # Écrire le fichier
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))
        
        print(f"Fichier Markdown généré : {output[0]}")
               

rule clean:
    shell:
        """
        rm -rf maps stats results tmp images
        """        

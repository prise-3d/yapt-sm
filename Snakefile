import pandas as pd

import os

# Configuration
configfile: "config.yaml"

# Load paramètre CSV
# 
# A lancer avec :
# snakemake --core 4 --config params_file=params.csv
df = pd.read_csv(config["params_file"], index_col="name")
df.columns = df.columns.str.strip()  # Nettoyage des espaces dans les noms de colonnes
params = df.T.to_dict()

# Extraire les fonctions depuis la colonne 'source' (en enlevant les extensions)
FUNCTIONS = sorted(set(os.path.splitext(os.path.basename(src))[0] for src in df["source"]))
print("Fonctions trouvées :", FUNCTIONS)


rule all:
    input:
        "results/stats_summary.csv",
        "results/tables.md",
        expand("results/plots_by_function/plot_variance_{function}.pdf", function=FUNCTIONS),
        expand("results/plots_by_function/plot_time_{function}.pdf", function=FUNCTIONS),
        expand("results/plots_by_function/plot_link_{function}.pdf", function=FUNCTIONS)
        

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
        """
        set -x
        mkdir -p {params.tmpdir}
        mkdir -p {config[output_image_path]}
        mkdir -p results/times

        # Measure wall-clock time in seconds and store it
        /usr/bin/time -f "%e" -o {output.timefile} {params.yapt}/yapt width=100 threads=1 dir={params.tmpdir} {params.args}

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

            # ajouter une colonne "precision" calculée comme (1 / (stddev * stddev))       
            stats["precision"] = 1 / (stats["stddev"] * stats["stddev"]) if stats["stddev"] != 0 else float("nan")

        df = pd.DataFrame(rows)
        df = df[["name", "source", "spp", "min", "max", "avg", "stddev", "time", "aggregator", "sampler", "precision"]]
        df.to_csv(output[0], index=False)


rule plot_variance:
    input:
        stats="results/stats_summary.csv",
        params=config.get("params_file", "params.csv")
    output:
        "results/plots_by_function/plot_variance_{function}.pdf"
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
        df_params = pd.read_csv(input.params)
        df_params.columns = df_params.columns.str.strip()
        extra_cols = [col for col in df_params.columns if col != "name" and col not in df_stats.columns]
        df = pd.merge(df_stats, df_params[["name"] + extra_cols], on="name")
        df.columns = df.columns.str.strip()

        # Filter by function name
        df_filtered = df[df["source"] == function_name]
        if df_filtered.empty:
            df_filtered = df[df["source"].str.contains(function_name, case=False, na=False)]

        df_filtered = df_filtered[df_filtered["spp"] > 10].sort_values("spp")

        if df_filtered.empty:
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
        mc_df = df[df["aggregator"] == "mc"]
        other_aggs = sorted(set(df["aggregator"]) - {"mc"})

        if not other_aggs:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            for aggregator, sub_group in mc_df.groupby("aggregator"):
                sub_group = sub_group.sort_values("spp")
                sns.lineplot(ax=ax1, x="spp", y="avg", data=sub_group, marker="o", label=f"{aggregator}")
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
            fig.suptitle(f"Function: {function_name} (mc only)", fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            n = len(other_aggs)
            cols = 2
            rows = math.ceil(n / cols)
            fig = plt.figure(figsize=(6 * cols, 4 * rows))
            for i, aggregator in enumerate(other_aggs):
                ax = plt.subplot(rows, cols, i + 1)
                agg_df = df[df["aggregator"] == aggregator]
                for aggregator, sub_group in agg_df.groupby("aggregator"):
                    sub_group = sub_group.sort_values("spp")
                    sns.lineplot(ax=ax, x="spp", y="avg", data=sub_group, marker="o", label=f"{aggregator}")
                    ax.fill_between(sub_group["spp"],
                                    sub_group["avg"] - sub_group["stddev"],
                                    sub_group["avg"] + sub_group["stddev"],
                                    alpha=0.3)
                fig_title = aggregator                                    
                if not mc_df.empty:
                    for aggregator, sub_group in mc_df.groupby("aggregator"):
                        sub_group = sub_group.sort_values("spp")
                        sns.lineplot(ax=ax, x="spp", y="avg", data=sub_group, linestyle="--", marker="x", label=f"{aggregator}")
                        ax.fill_between(sub_group["spp"],
                                        sub_group["avg"] - sub_group["stddev"],
                                        sub_group["avg"] + sub_group["stddev"],
                                        alpha=0.15)
                ax.set_title(f"{fig_title} vs MC")
                ax.set_xscale("log")
                ax.set_xlabel("Number of samples")
                ax.set_ylabel("Mean")
                ax.grid(True)
                ax.legend()                
            fig.suptitle(f"Dispersion of Voronoi and MC Estimators", fontsize=14)
            #fig.suptitle(f"Performance comparison  {function_name}\n(mc shown on all)", fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        fig.savefig(output[0], dpi=150, bbox_inches='tight')
        plt.close(fig)

rule plot_time:
    input:
        stats="results/stats_summary.csv",
        params=config.get("params_file", "params.csv")
    output:
        "results/plots_by_function/plot_time_{function}.pdf"
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
        df_params = pd.read_csv(input.params)
        df_params.columns = df_params.columns.str.strip()
        extra_cols = [col for col in df_params.columns if col != "name" and col not in df_stats.columns]
        df = pd.merge(df_stats, df_params[["name"] + extra_cols], on="name")
        df.columns = df.columns.str.strip()

        # Filter by function name
        df_filtered = df[df["source"] == function_name]
        if df_filtered.empty:
            df_filtered = df[df["source"].str.contains(function_name, case=False, na=False)]

        df_filtered = df_filtered[df_filtered["spp"] > 10].sort_values("spp")

        if df_filtered.empty:
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
        mc_df = df[df["aggregator"] == "mc"]
        other_aggs = sorted(set(df["aggregator"]) - {"mc"})

        if not other_aggs:
            fig, ax2 = plt.subplots(figsize=(10, 6))
            for aggregator, sub_group in mc_df.groupby("aggregator"):
                sub_group = sub_group.sort_values("spp")
                sns.lineplot(ax=ax2, x="spp", y="time", data=sub_group, marker="o", label=f"{aggregator}")
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
            fig = plt.figure(figsize=(6 * cols, 4 * rows))
            for i, aggregator in enumerate(other_aggs):
                ax = plt.subplot(rows, cols, i + 1)
                agg_df = df[df["aggregator"] == aggregator]
                for aggregator, sub_group in agg_df.groupby("aggregator"):
                    sub_group = sub_group.sort_values("spp")
                    sns.lineplot(ax=ax, x="spp", y="time", data=sub_group, marker="o", label=f"{aggregator}")
                if not mc_df.empty:
                    for aggregator, sub_group in mc_df.groupby("aggregator"):
                        sub_group = sub_group.sort_values("spp")
                        sns.lineplot(ax=ax, x="spp", y="time", data=sub_group, linestyle="--", marker="x", label=f"{aggregator}")
                ax.set_title(f"Execution Time - Aggregator: {aggregator}")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("Samples per pixel (SPP)")
                ax.set_ylabel("Time (seconds)")
                ax.grid(True)
                ax.legend()
            fig.suptitle(f"Execution Time vs SPP by aggregator — Function: {function_name}\n(mc shown on all)", fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        fig.savefig(output[0], dpi=150, bbox_inches='tight')
        plt.close(fig)

rule plot_link:
    input:
        stats="results/stats_summary.csv",
        params=config.get("params_file", "params.csv")
    output:
        "results/plots_by_function/plot_link_{function}.pdf"

    run:
        print(f"→ Generating time vs variance plot for function: {wildcards.function}") 
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
        df_params = pd.read_csv(input.params)
        df_params.columns = df_params.columns.str.strip()
        extra_cols = [col for col in df_params.columns if col != "name" and col not in df_stats.columns]
        df = pd.merge(df_stats, df_params[["name"] + extra_cols], on="name")
        df.columns = df.columns.str.strip()

        # Filter by function name
        df_filtered = df[df["source"] == function_name]
        if df_filtered.empty:
            df_filtered = df[df["source"].str.contains(function_name, case=False, na=False)]

        df_filtered = df_filtered[df_filtered["spp"] > 10].sort_values("spp")

        if df_filtered.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Aucune donnée disponible\npour la fonction '{function_name}'", 
                   ha='center', va='center', fontsize=16, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_title(f"Fonction: {function_name}", fontsize=14)
            ax.axis('off')
        else:
            df_filtered["variance"] = df_filtered["stddev"] * df_filtered["stddev"]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"Time vs Variance - Function: {function_name}")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Variance")
            ax.set_xscale("log")
            ax.grid(True)
            for aggregator, sub_group in df_filtered.groupby("aggregator"):
                sub_group = sub_group.sort_values("time")
                sns.lineplot(ax=ax, x="time", y="variance", data=sub_group, marker="o", label=f"{aggregator}")

        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        fig.savefig(output[0], dpi=150, bbox_inches='tight')
        plt.close(fig)
  

rule generate_markdown_tables:
    input:
        stats="results/stats_summary.csv",
        params=config.get("params_file", "params.csv")
    output:
        "results/tables.md",
        "results/tables_summary.md"
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

        # Début du document Markdown complet
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

        # Écrire le fichier complet
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))

        print(f"Fichier Markdown généré : {output[0]}")

        # Génération du résumé tables_summary.md
        summary_lines = [
            "# Résumé des variances (StdDev) pour le SPP maximal",
            "",
            "| Fonction | StdDev MC | StdDev Vor |",
            "|----------|-----------|------------|"
        ]

        for function_name in functions:
            df_func = df[df["source"] == function_name]
            if df_func.empty:
                continue

            # Trouver le SPP maximal pour cette fonction
            max_spp = df_func["spp"].max()
            df_max_spp = df_func[df_func["spp"] == max_spp]

            # MC
            stddev_mc = df_max_spp[df_max_spp["aggregator"] == "mc"]["stddev"]
            stddev_mc_val = f"{stddev_mc.values[0]:.6f}" if not stddev_mc.empty else ""
            time_mc = df_max_spp[df_max_spp["aggregator"] == "mc"]["time"]
            time_mc_val = f"{time_mc.mean():.3f}" if not time_mc.empty else ""

            # Vor
            stddev_vor = df_max_spp[df_max_spp["aggregator"] == "vor"]["stddev"]
            stddev_vor_val = f"{stddev_vor.values[0]:.6f}" if not stddev_vor.empty else ""
            time_vor = df_max_spp[df_max_spp["aggregator"] == "vor"]["time"]
            time_vor_val = f"{time_vor.mean():.3f}" if not time_vor.empty else ""

            # CVor
            stddev_cvor = df_max_spp[df_max_spp["aggregator"] == "cvor"]["stddev"]
            stddev_cvor_val = f"{stddev_cvor.values[0]:.6f}" if not stddev_cvor.empty else ""
            time_cvor = df_max_spp[df_max_spp["aggregator"] == "cvor"]["time"]
            time_cvor_val = f"{time_cvor.mean():.3f}" if not time_cvor.empty else ""

            # FVor
            stddev_fvor = df_max_spp[df_max_spp["aggregator"] == "fvor"]["stddev"]
            stddev_fvor_val = f"{stddev_fvor.values[0]:.6f}" if not stddev_fvor.empty else ""
            time_fvor = df_max_spp[df_max_spp["aggregator"] == "fvor"]["time"]
            time_fvor_val = f"{time_fvor.mean():.3f}" if not time_fvor.empty else ""

            summary_lines.append(f"| {function_name} | {stddev_mc_val} | {stddev_vor_val} | {stddev_cvor_val} | {stddev_fvor_val} |{time_mc_val} | {time_vor_val} | {time_cvor_val} | {time_fvor_val} | ")

        # Update header for new columns
        summary_lines[2] = "| Fonction | StdDev MC | StdDev Vor | StdDev CVor | StdDev FVor | Time MC | Time Vor | Time CVor | Time FVor |"
        summary_lines[3] = "|----------|-----------|------------|-------------|-------------|---------|----------|-----------|-----------|"

        with open(output[1], 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
               

rule clean:
    shell:
        """
        rm -rf maps stats results tmp images
        """        

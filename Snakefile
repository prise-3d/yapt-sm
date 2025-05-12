import pandas as pd

FUNCTIONS = ["unit_disk", "f1", "f2", "disturbed_disk"]

# Configuration
configfile: "config.yaml"

# load paramters in the CSV
df = pd.read_csv(config["params_file"], index_col="name")
df.columns = df.columns.str.strip()  # delete spaces in column names
params = df.T.to_dict()

# path to yapt
yapt_path = lambda wildcards, config: config["yapt_path"]


rule all:
    input:
        "results/stats_summary.csv",
        "results/plots_by_function/plot_unit_disk.png",
        "results/plots_by_function/plot_f1.png",
        "results/all_plots.pdf",
        "results/stats_summary.csv",
        "results/plots_by_function/plot_unit_disk.png",
        "results/plots_by_function/plot_f1.png",
        "results/plots_by_function/plot_f2.png",
        "results/plots_by_function/plot_disturbed_disk.png"
        

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
        expand("stats/{name}.txt", name=params.keys())
    output:
        "results/stats_summary.csv"
    params:
        all_params=params
    run:
        import re
        import pandas as pd

        rows = []
        for infile in input:
            name = infile.split("/")[-1].replace(".txt", "")
            param = params["all_params"][name]

            with open(infile) as f:
                content = f.read()

            stats = {}
            for stat in ["Min", "Max", "Avg", "StdDev"]:
                match = re.search(rf"{stat}:\s+([0-9.eE+-]+)", content)
                stats[stat.lower()] = float(match.group(1)) if match else float("nan")

            stats["name"] = name
            stats["spp"] = param["spp"]
            stats["source"] = os.path.splitext(os.path.basename(param["source"]))[0]
            stats["aggregator"] = param["aggregator"]
            stats["sampler"] = param["sampler"]
            rows.append(stats)            

        df = pd.DataFrame(rows)
        df = df[["name", "source", "spp", "min", "max", "avg", "stddev", "aggregator", "sampler"]]
        df.to_csv(output[0], index=False)


rule plot_stats:
    input:
        stats="results/stats_summary.csv",
        params="params.csv"
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
        df_params = pd.read_csv(input.params)
        df_params.columns = df_params.columns.str.strip()
        extra_cols = [col for col in df_params.columns if col != "name" and col not in df_stats.columns]
        df = pd.merge(df_stats, df_params[["name"] + extra_cols], on="name")
        df.columns = df.columns.str.strip()

        # Filter by function name and spp
        df = df[df["source"].str.contains(function_name)]
        df = df[df["spp"] > 10].sort_values("spp")

        if df.empty:
            raise ValueError(f"Aucune donnée pour la fonction '{function_name}'")

        # Séparer les agrégateurs
        mc_df = df[df["aggregator"] == "mc"]
        other_aggs = sorted(set(df["aggregator"]) - {"mc"})
        n = len(other_aggs)
        cols = 2
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

        for i, aggregator in enumerate(other_aggs):
            row, col = divmod(i, cols)
            ax = axes[row][col]

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

            # Courbes pour 'mc' en référence
            for sampler, sub_group in mc_df.groupby("sampler"):
                sub_group = sub_group.sort_values("spp")
                sns.lineplot(ax=ax, x="spp", y="avg", data=sub_group, linestyle="--", marker="x", label=f"mc/{sampler}")

                ax.fill_between(sub_group["spp"],
                                sub_group["avg"] - sub_group["stddev"],
                                sub_group["avg"] + sub_group["stddev"],
                                alpha=0.15)

            ax.set_title(f"Aggregator: {aggregator}")
            ax.set_xscale("log")
            ax.set_xlabel("Samples per pixel (SPP)")
            ax.set_ylabel("Mean")
            ax.grid(True)
            ax.legend()

        # Supprimer les axes non utilisés
        for i in range(n, rows * cols):
            fig.delaxes(axes[i // cols][i % cols])

        fig.suptitle(f"Mean vs SPP by Sampler — Function: {function_name}\n(mc shown on all)", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        fig.savefig(output[0])
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

rule clean:
    shell:
        """
        rm -rf maps stats results tmp images
        """        

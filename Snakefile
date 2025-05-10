import pandas as pd

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
        "results/all_plots.pdf"

# Rule to run yapt and process .exr files
rule run_yapt:
    output:
        mapfile="maps/{name}.txt"
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

        function_name = wildcards.function

        # Load and merge
        df_stats = pd.read_csv(input.stats)
        df_params = pd.read_csv(input.params)
        df_params.columns = df_params.columns.str.strip()
        extra_cols = [col for col in df_params.columns if col != "name" and col not in df_stats.columns]
        df = pd.merge(df_stats, df_params[["name"] + extra_cols], on="name")
        df.columns = df.columns.str.strip()

        # Filter by function name
        df = df[df["source"].str.contains(function_name)]

        if df.empty:
            raise ValueError(f"No data found for function '{function_name}'")

        df = df.sort_values("spp")
        df = df[df["spp"] > 10]

        plt.figure(figsize=(10, 5))
        for aggregator, sub_group in df.groupby("aggregator"):
            sub_group = sub_group.sort_values("spp")
            sns.lineplot(x="spp", y="avg", data=sub_group, marker="o", label=aggregator)
            plt.fill_between(sub_group["spp"],
                             sub_group["avg"] - sub_group["stddev"],
                             sub_group["avg"] + sub_group["stddev"],
                             alpha=0.3)

        plt.title(f"Mean vs SPP by Aggregator\nFunction: {function_name}")
        plt.xlabel("Samples per pixel (SPP)")
        plt.ylabel("Mean")
        plt.grid(True)
        plt.legend()
        plt.xscale("log")
        plt.tight_layout()

        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        plt.savefig(output[0])
        plt.close()

import glob
import matplotlib.pyplot as plt  # Added import for plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image        

rule combine_plots:
    input:
        png_files=glob.glob("results/plots_by_function/*.png")
    output:
        "results/all_plots.pdf"
    run:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
        from PIL import Image
        import os

        # Création du PDF dans le répertoire 'results'
        pdf_path = output[0]
        with PdfPages(pdf_path) as pdf:
            for img_path in input.png_files:
                image = Image.open(img_path)
                fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100))
                ax.imshow(image)
                ax.axis("off")  # Supprimer les axes
                pdf.savefig(fig)
                plt.close(fig)                

rule clean:
    shell:
        """
        rm -rf maps stats results tmp images
        """        
import pandas as pd

# Configuration
configfile: "config.yaml"

# load paramters in the CSV
df = pd.read_csv("params.csv", index_col="name")
df.columns = df.columns.str.strip()  # delete spaces in column names
params = df.T.to_dict()

# path to yapt
yapt_path = lambda wildcards, config: config["yapt_path"]


rule all:
    input:
        "results/stats_summary.csv",
        "results/stats_plot.png"

# Rule to run yapt and process .exr files
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
            rows.append(stats)
            print(f"→ {name}: Min={stats['min']}, Max={stats['max']}, Avg={stats['avg']}, StdDev={stats['stddev']}", file=sys.stderr)

        df = pd.DataFrame(rows)
        df = df[["name", "spp", "min", "max", "avg", "stddev"]]
        df.to_csv(output[0], index=False)


rule plot_stats:
    input:
        "results/stats_summary.csv"
    output:
        "results/stats_plot.png"
    run:
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        df = pd.read_csv(input[0])

        # Filtrer uniquement la colonne 'avg'
        df = df[["spp", "avg"]].sort_values("spp")

        plt.figure(figsize=(10, 5))
        sns.lineplot(x="spp", y="avg", data=df, marker="o")
        plt.title("Évolution de la moyenne en fonction du SPP")
        plt.ylabel("Moyenne")
        plt.xlabel("Samples per pixel (SPP)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output[0])
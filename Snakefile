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
                    # Search for the first number after "Avg:"
                    match = re.search(r"Avg:\s+([0-9.eE+-]+)", content)
                    mean = match.group(1) if match else "NaN"
                    out.write(f"{name},{mean}\n")

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
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 4))
        ax = sns.barplot(x="name", y="mean", data=df)
        ax.set_title("Moyenne par test")
        ax.set_ylabel("Mean")
        ax.set_xlabel("Test")
        plt.tight_layout()
        plt.savefig(output[0])

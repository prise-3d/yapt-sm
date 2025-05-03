import pandas as pd

# Configuration
configfile: "config.yaml"

# Charger les paramètres du CSV
params = pd.read_csv("params.csv", index_col="name").T.to_dict()

# Chemin vers yapt
yapt_path = lambda wildcards, config: config["yapt_path"]

rule all:
    input:
        "results/stats_summary.csv"

rule run_yapt:
    output:
        "images/{name}.exr" 
    params:
        yapt=config["yapt_path"],
        dir=config["output_image_path"],
        args=lambda wildcards: "source={}/{}".format(config["function_path"], params[wildcards.name]["source"])
    shell:
        "{params.yapt}/yapt path={output} {params.args}"

rule extract_stats:
    input:
        lambda wildcards: f"{config['output_image_path']}/{wildcards.name}.exr"
    output:
        "stats/{name}.txt"
    shell:
        "oiiotool --stats {input} | grep 'Mean' > {output}"

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
                    match = re.search(r"Mean:\s+([0-9.eE+-]+)", content)
                    mean = match.group(1) if match else "NaN"
                    out.write(f"{name},{mean}\n")

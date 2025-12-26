from config import config
import os
import pandas as pd
import json
from data_loader import data_loader
from join_discoverer import JoinDiscoverer
from openai import OpenAI
from config import config


if __name__ == "__main__":
    if os.path.exists(config.local_output_directory + "stats_and_semantics_total.json"):
        with open(config.local_output_directory + "stats_and_semantics_total.json", "r") as f:
            data_lake = json.load(f)
    else:
        data_lake = data_loader.load_semantics_and_stats()
        if data_lake:
            print(config.add_timestamp(config.color_text("Semantics and statistics loaded successfully.", "green")))
        else:
            print(config.add_timestamp(config.color_text("Failed to load semantics and statistics.", "red")))

    print(config.add_timestamp(config.color_text(f"Total tables in data lake: {len(data_lake)}", "green")))

    openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    if openai_client:
        print(config.add_timestamp(config.color_text("OpenAI client initialized successfully.", "green")))
    else:
        print(config.add_timestamp(config.color_text("Failed to initialize OpenAI client.", "red")))

    join_discoverer = JoinDiscoverer(openai_client)
    embeddings = join_discoverer.generate_embeddings(data_lake_list=data_lake)
    df_distances, df_joinable, evaluation_metrics = join_discoverer.compute_distances_and_evaluation(embeddings=embeddings)
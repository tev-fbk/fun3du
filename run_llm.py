import json

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

import ollama
from utils.sun3d.data_parser import DataParser
from utils.misc import sort_alphanumeric

OLLAMA_PORT = 11434


def get_LLM_response(statement, query):
    client = ollama.Client(host=f"localhost:{OLLAMA_PORT}")
    response = client.chat(
        model="llama3.1",
        messages=[
            {
                "role": "system",
                "content": "You are an AI System that has to provide json files to a robotic system so that it can interact with our physical world, based on a natural language prompt.\nIn particular, you have to help the robot in identify which object parts it has to interact with to solve particual tasks.\n its set of possible actions are [rotate, key_press, tip_push, hook_pull, pinch_pull, hook_turn, foot_push, plug_in, unplug]",
            },
            {
                "role": "user",
                "content": statement.format(query=query),
            },
        ],
        options={"temperature": 0},
    )

    return response["message"]["content"], response


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(args: DictConfig):

    statement = """How do I {query}?
respond directly with only the json with the following format.
{{
"prompt": a string with the prompt,
"task_solving_sequence": a list of strings with the description of what I have to do to accomplish the task described by the prompt, subdivided in subtasks,
"acted_on_object": a string with the name of the object part on which I have to act
"acted_on_object_hierarchy": a list of object parts from the top level object to the object part,
}}

"""
    parser = DataParser(args.dataset.root, args.dataset.split)
    visits = sort_alphanumeric(parser.get_visits())

    start = int(args.dataset.start)
    start = 0 if args.dataset.start is None else int(args.dataset.start)
    end = len(visits) if args.dataset.end is None else int(args.dataset.end)

    print(
        f"LLM processing for {end-start} visits (split {args.dataset.split}), from {visits[0]} to {visits[-1]}"
    )

    for visit_id in tqdm(visits):

        descs = parser.get_descriptions_list(visit_id)
        json_data = []

        for desc_id, query in descs.items():
            response = get_LLM_response(statement=statement, query=query)
            try:
                if '```' in response[0]:
                    json_dict = json.loads(response[0].split('```')[1])
                else:
                    json_dict = json.loads(response[0])
                json_data.append(json_dict)
            except Exception as e:
                print(f"error {e}")
                json_data.append({})

        new_path = f"{args.dataset.root}/{args.dataset.split}/{visit_id}/{visit_id}_{args.llm_type}_cot.json"

        with open(new_path, "w") as out_f:
            json.dump(json_data, out_f, indent=4)


if __name__ == "__main__":
    main()

import openai
import json
import argparse
import tqdm
import time

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    # argparser.add_argument('--prompt_fp', type=str, default='./prompt/readability_detailed.txt')
    argparser.add_argument('--save_fp', type=str, default=f'./results/result_readability_gpt4preview.json')
    argparser.add_argument('--data_fp', type=str, default='./data/data_final.json')
    #argparser.add_argument('--key', type=str, required=True)
    argparser.add_argument('--model', type=str, default='gpt-4-1106-preview')
    argparser.add_argument('--response_model', type=str, default='gpt-3.5')
    argparser.add_argument('--ability', type=str, default='readability')
    args = argparser.parse_args()
    #openai.api_key = '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'
    openai.api_key = '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'

    prompt_path = f'./prompt/{args.ability}_detailed.txt'
    data = json.load(open(args.data_fp))
    prompt = open(prompt_path).read()

    ct, ignore = 0, 0

    new_json = []
    for instance in tqdm.tqdm(data):
        source = instance['question']
        system_output = instance[args.response_model]['response']
        cur_prompt = prompt.replace('{{Question}}', source).replace('{{Response}}', system_output)
        #instance['prompt'] = cur_prompt
        while True:
            try:
                _response = openai.ChatCompletion.create(
                    model=args.model,
                    messages=[{"role": "system", "content": cur_prompt}],
                    temperature=2,
                    max_tokens=3,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    # logprobs=40,
                    n=5,
                    request_timeout = 30
                )
                time.sleep(1.5)

                all_responses = [_response['choices'][i]['message']['content'] for i in
                                 range(len(_response['choices']))]
                instance[args.response_model][args.ability] = all_responses
                new_json.append(instance)
                ct += 1
                break
            except Exception as e:
                print(e)
                if ("limit" in str(e)):
                    time.sleep(2)
                else:
                    ignore += 1
                    print('ignored', ignore)
                    new_json.append(instance)

                    break

    print('ignored total', ignore)
    with open(args.save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)
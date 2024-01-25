import argparse
import os
import re
import json
import random
import torch
from tqdm import tqdm
from model.rl_models import CQLModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from eval.utils import generate_completions, load_hf_lm_and_tokenizer, rl_generate_completions
from eval.gsm.math_tool import *
from eval.gsm.utils import *
from eval.utils import lumos_dir_path, test_model_path


def rl_execute(action, inter_results):
    action_variable, action_name, action_args = parse_actions(action)
    variables = findvariables(action_args)

    if action_name in ["", "Calculator", "SetEquation", "SolveEquation", "Define"]:
        updated_action_args = action_args
        exist_ops = examine_ops(action_args)
        if isfunc(action_args) or not exist_ops:
            try:
                for variable in variables:
                    updated_action_args = updated_action_args.replace(variable, str(inter_results[variable]))
            except:
                return None, None
        else:
            try:
                for variable in variables:
                    if '=' in str(inter_results[variable]):
                        updated_action_args = updated_action_args.replace(variable, f":({str(inter_results[variable])}):")
                    else:
                        updated_action_args = updated_action_args.replace(variable, f"({str(inter_results[variable])})")
            except:
                return None, None
        
        op_result = '1'
        try:
            if action_name == "Calculator":
                return action_variable[0], op_result
            elif action_name == "SetEquation":
                return action_variable[0], op_result
            elif action_name == "SolveEquation":
                eq_results = [op_result] * len(action_variable)
                return action_variable, eq_results
            elif action_name == "Define":
                return action_variable[0], op_result
            elif action_name == "":
                return action_variable[0], updated_action_args
        except:
            return None, None
        
    return None, None


def execute(action, inter_results):
    action_variable, action_name, action_args = parse_actions(action)
    variables = findvariables(action_args)

    if action_name in ["", "Calculator", "SetEquation", "SolveEquation", "Define"]:
        updated_action_args = action_args
        exist_ops = examine_ops(action_args)
        if isfunc(action_args) or not exist_ops:
            try:
                for variable in variables:
                    updated_action_args = updated_action_args.replace(variable, str(inter_results[variable]))
            except:
                return None, None
        else:
            try:
                for variable in variables:
                    if '=' in str(inter_results[variable]):
                        updated_action_args = updated_action_args.replace(variable, f":({str(inter_results[variable])}):")
                    else:
                        updated_action_args = updated_action_args.replace(variable, f"({str(inter_results[variable])})")
            except:
                return None, None
        
        try:
            if action_name == "Calculator":
                return action_variable[0], MathActions.Calculator(updated_action_args, isfunc(action_args))
            elif action_name == "SetEquation":
                return action_variable[0], MathActions.SetEquation(updated_action_args)
            elif action_name == "SolveEquation":
                eq_results = MathActions.SolveEquation(updated_action_args)
                if '±' in eq_results:
                    terms = eq_results.split('±')
                    eq_results = f"{terms[0]} + {terms[1]}"
                eq_results = eq_results.split(" ∧ ")
                for i in range(len(eq_results)):
                    pos_eq = eq_results[i].find('=')
                    eq_results[i] = eq_results[i][pos_eq + 1:]
                return action_variable, eq_results
            elif action_name == "Define":
                return action_variable[0], MathActions.Define(updated_action_args)
            elif action_name == "":
                return action_variable[0], updated_action_args
        except:
            return None, None


def lumos_iterative(args, infer_context: dict):
    random.seed(42)

    print("Loading data...")
    test_data, subgoals, actions, plan_prompts, ground_prompts, all_subgoals_actions = list(), list(), list(), list(), list(), list()
    with open(os.path.join(args.data_dir, f"test.jsonl")) as fin:
        for line in fin:
            example = json.loads(line)
            test_data.append({
                "question": example["question"],
                "answer": example["answer"].split("####")[1].strip()
            })
        
    # some numbers are in the `x,xxx` format, and we want to remove the comma
    for example in test_data:
        example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["answer"])
        assert float(example["answer"]), f"answer is not a valid number: {example['answer']}"

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = test_data[:args.max_num_examples]
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    solved_idx = list()
    for i in tqdm(range(16)):
        if len(solved_idx) == len(test_data):
            break
        if i % 2 == 0:
            prompt_prefix = "Please provide a reasonable subgoal-based plan to solve the given task.\n"
        else:
            prompt_prefix = "Please ground the given subgoal to corresponding executable actions for solving the given task. The grounded actions must be the one in available action list.\n\n" \
                            "The available action list is 'Calculator', 'SetEquation', 'SolveEquation', 'Count', 'SolveInequality', 'Code', and 'Define'.\n" \
                            "Calculator(formula): Calculate the input formula; SetEquation(equation): Set up an equation to be solved; SolveEquation(equation): Solve the previous set equation; Count(list): Count the number of elements in the given list; SolveInequality(inequality): Solve the previous set inequality; Code(pseudo_code): Generate a Python function that corresponds to the pseudo code; Define(variable/number): Define a variable or a number for latter usage.\n\n"
        
        assert not subgoals or len(subgoals) == len(test_data)
        assert not actions or len(actions) == len(test_data)
        
        if i % 2 == 0:
            for j, example in enumerate(test_data):
                if i == 0:
                    all_subgoals_actions.append({"question": example["question"].strip(), "answer": example["answer"], "subgoals": [], "actions": [], "results": dict()})
                    plan_prompt = "<|user|>\n" + prompt_prefix + "Task: " + example["question"].strip() + "; Initial Environment Description: None.\n<|assistant|>\n"
                    plan_prompts.append(plan_prompt)
                else:
                    plan_prompts[j] += subgoals[j].strip() + f"\n\n<|user|>\nThe executed result for Subgoal {str(i // 2)} is {all_subgoals_actions[j]['actions'][-1].split('=')[-1].strip()}. Should we stop planning?\n<|assistant|>\n"
        else:
            for j, example in enumerate(test_data):
                if i == 1:
                    ground_prompt = "<|user|>\n" + prompt_prefix + "Task: " + example["question"].strip() + f" \nSubgoal to be grounded: {subgoals[j].strip()}\n<|assistant|>\n"
                    ground_prompts.append(ground_prompt)
                else:
                    ground_prompts[j] += actions[j].strip() + f"\n\n<|user|>\nSubgoal to be grounded: {subgoals[j].strip()}\n<|assistant|>\n"

        if args.model_name_or_path:
            print("Loading model and tokenizer...")
            if i % 2 == 0:
                model = infer_context['plan_model']
                tokenizer = infer_context['plan_tokenizer']
            else:
                model = infer_context['ground_model']
                tokenizer = infer_context['ground_tokenizer']
            
            new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.
            if i % 2 == 0:
                subgoals = generate_completions(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=plan_prompts,
                    max_new_tokens=512,
                    batch_size=args.eval_batch_size,
                    stop_id_sequences=[[new_line_token]]
                )
            else:
                if isinstance(model, CQLModelForCausalLM):
                    actions = rl_generate_completions(
                        rl_model=model,
                        tokenizer=tokenizer,
                        prompts=ground_prompts,
                        max_new_tokens=512,
                        batch_size=args.eval_batch_size,
                        stop_id_sequences=[[new_line_token]]
                    )
                else:
                    actions = generate_completions(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=ground_prompts,
                        max_new_tokens=512,
                        batch_size=args.eval_batch_size,
                        stop_id_sequences=[[new_line_token]]
                    )
        
        if i % 2 == 0:
            for j in range(len(test_data)):
                if j in solved_idx:
                    continue
                if "Yes, I will stop planning" in subgoals[j].strip():
                    solved_idx.append(j)
        else:
            for j in range(len(test_data)):
                if j in solved_idx or all_subgoals_actions[j]["results"] is None:
                    continue
                all_subgoals_actions[j]["subgoals"].append(subgoals[j].strip())
                for k, action in enumerate(actions[j].strip().split('; ')):
                    try:
                        results_variable, execution_results = execute(action.strip(), all_subgoals_actions[j]["results"])
                        if execution_results:
                            if isinstance(results_variable, str):
                                all_subgoals_actions[j]["results"][results_variable] = execution_results
                            else:
                                for k, variable in enumerate(results_variable):
                                    all_subgoals_actions[j]["results"][variable] = execution_results[k]
                        else:
                            all_subgoals_actions[j]["results"] = None
                    except:
                        all_subgoals_actions[j]["results"] = None
                
                    try:
                        if all_subgoals_actions[j]["results"]:
                            if isinstance(execution_results, str):
                                all_subgoals_actions[j]["actions"].append(action.strip() + " = " + execution_results)
                            else:
                                all_subgoals_actions[j]["actions"].append(action.strip() + " = " + ", ".join(execution_results))
                        else:
                            all_subgoals_actions[j]["actions"].append(action.strip())
                    except:
                        all_subgoals_actions[j]["actions"].append(action.strip())
    
    corr = 0
    with open(os.path.join(args.save_dir, f"predictions_iterative.jsonl"), "w") as f:
        for subgoal_action in all_subgoals_actions:
            if subgoal_action["results"]:
                variables = sorted(list([int(k[1:]) for k in subgoal_action["results"].keys()]))
                final_variable = 'R' + str(variables[-1])
                if subgoal_action["results"][final_variable] == subgoal_action["answer"]:
                    corr += 1
                
                f.write(json.dumps({"question": subgoal_action["question"], 
                                    "pred": subgoal_action["results"][final_variable], 
                                    "answer": subgoal_action["answer"], 
                                    "subgoals": subgoal_action["subgoals"], 
                                    "actions": subgoal_action["actions"]
                                    })+'\n')
    print("Acc:", corr*1./len(test_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", type=str)
    # parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate.")
    # parser.add_argument("--save_dir", type=str)
    # parser.add_argument("--model_name_or_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    # parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    # parser.add_argument("--formulation", type=str, default='lumos_iterative', help="considered formulation.")
    # parser.add_argument("--load_in_8bit", action="store_true", help="load model in 8bit mode, which will reduce memory and speed up inference.")
    # parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")

    parser.add_argument("--debug", action="store_true", help="debug flag")
    parser.add_argument("--frozen_attn", action="store_true", help="frozen attn flag")
    parser.add_argument("--low_cpu_mem_usage", action="store_true", default=False, help=("It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. If passed, LLM loading time and RAM consumption will be benefited."))
    parser.add_argument("--use_flash_attn", action="store_true", default=False, help="If passed, will use flash attention to train the model.")
    parser.add_argument("--use_slow_tokenizer", action="store_false", default=True, help="If passed, will use flash attention to train the model.")

    parser.add_argument("--data_dir", type=str, default=lumos_dir_path.joinpath('data/eval/gsm'), help="data directory.")
    parser.add_argument("--max_num_examples", type=int, default=10, help="maximum number of examples to evaluate.")
    parser.add_argument("--save_dir", type=str, default=lumos_dir_path.joinpath('results/gsm'), help="directory to save the results.")
    parser.add_argument("--model_name_or_path", type=str, default=test_model_path, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--formulation", type=str, default='lumos_iterative', help="considered formulation.")
    parser.add_argument("--load_in_8bit", action="store_true", help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    args = parser.parse_args()

    if args.debug:
        args.use_flash_attn = True
        args.use_slow_tokenizer = True
        args.low_cpu_mem_usage = True  # Used by device_map = 'auto'
    # Train config
    else:
        args.use_flash_attn = True
        args.use_slow_tokenizer = True

    plan_model_path = '/home/ubuntu/lumos/.cache/hub/models--ai2lumos--lumos_maths_plan_iterative/snapshots/232661635c70cd18d1cd0c3acad7ea9325e435bf'
    # plan_model_path = '/home/ubuntu/lumos/.cache/hub/models--ai2lumos--lumos_maths_ground_iterative/snapshots/edd152df62ff0c1f4e6297ed83fc7ade62bf6c80'
    plan_model, plan_tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=plan_model_path, 
                tokenizer_name_or_path=plan_model_path, 
                load_in_8bit=args.load_in_8bit, 
                load_in_half=True,
                gptq_model=args.gptq
            )
    
    # save_directory = '/home/ubuntu/lumos/results/lumos_maths_rl_iterative/train/step_40'
    # load_context = CQLModelForCausalLM.prepare_load_context(args=args, save_directory=save_directory)
    # ground_model = CQLModelForCausalLM(**load_context)
    # ground_model.from_pretrained(args=args, save_directory=save_directory)
    # ground_tokenizer = ground_model.tokenizer
    # ground_model.to('cuda:1')

    ground_model_path = '/home/ubuntu/lumos/.cache/hub/models--ai2lumos--lumos_maths_ground_iterative/snapshots/edd152df62ff0c1f4e6297ed83fc7ade62bf6c80'
    ground_model, ground_tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=ground_model_path, 
                tokenizer_name_or_path=ground_model_path, 
                load_in_8bit=args.load_in_8bit, 
                load_in_half=True,
                gptq_model=args.gptq
            )

    infer_context = dict(
        plan_model=plan_model,
        plan_tokenizer=plan_tokenizer,
        ground_model=ground_model,
        ground_tokenizer=ground_tokenizer,
    )

    if args.formulation == "lumos_iterative":
        lumos_iterative(args, infer_context=infer_context)
    else:
        raise NotImplementedError

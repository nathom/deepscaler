# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


from deepscaler.rewards.math_reward import deepscaler_reward_fn

def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    else:
        return deepscaler_reward_fn


class RewardManager():
    """The reward manager.
    
    Handles reward computation for model responses, including:
    1. Processing standard responses
    2. Handling tool calls in responses
    3. Tracking tool usage metrics
    """

    def __init__(self, tokenizer, num_examine, config=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.config = config
        
        # Initialize tools if enabled in config
        self.tools_available = False
        
        if config and 'tools' in config and config.tools.enabled:
            try:
                from deepscaler.rewards.tool_utils import (
                    ToolRegistry, Tool, SearchTool, parse_tool_calls, insert_tool_responses
                )
                import importlib
                import json
                import os
                from deepscaler.system_prompts import TOOL_ENABLED_SYSTEM_PROMPT, DEEPSEEK_MATH_SYSTEM_PROMPT
                
                # Setup tool registry
                self.tool_registry = ToolRegistry()
                self.tools_to_use = config.tools.tools_to_use
                
                # Register requested tools
                if "search" in self.tools_to_use:
                    knowledge_base = {}
                    
                    # Load knowledge base if specified
                    if config.tools.knowledge_base_path:
                        kb_path = os.path.expanduser(config.tools.knowledge_base_path)
                        if os.path.exists(kb_path):
                            try:
                                with open(kb_path, 'r') as f:
                                    knowledge_base = json.load(f)
                                print(f"Loaded knowledge base from {kb_path} with {len(knowledge_base)} entries")
                            except Exception as e:
                                print(f"Error loading knowledge base: {e}")
                    
                    search_tool = SearchTool(knowledge_base=knowledge_base)
                    self.tool_registry.register_tool(search_tool)
                
                # Check if the wikisearch.py file exists
                wikisearch_path = os.path.expanduser("~/wikisearch/wikisearch.py")
                if not os.path.exists(wikisearch_path):
                    print(f"Warning: Wikisearch file not found at {wikisearch_path}")
                else:
                    print(f"Wikisearch file found at {wikisearch_path}")
                
                # Register Wikipedia tools if requested
                if "search_wikipedia_titles" in self.tools_to_use:
                    wiki_title_tool = WikipediaTitleSearchTool()
                    self.tool_registry.register_tool(wiki_title_tool)
                    
                if "search_wikipedia_content" in self.tools_to_use:
                    wiki_content_tool = WikipediaContentSearchTool()
                    self.tool_registry.register_tool(wiki_content_tool)
                    
                if "search_wikipedia_sections" in self.tools_to_use:
                    wiki_section_tool = WikipediaSectionSearchTool()
                    self.tool_registry.register_tool(wiki_section_tool)
                
                # For custom tools, import dynamically based on class name
                for tool_name in self.tools_to_use:
                    if tool_name != "search" and ":" in tool_name:
                        module_path, class_name = tool_name.split(":")
                        try:
                            module = importlib.import_module(module_path)
                            tool_class = getattr(module, class_name)
                            if issubclass(tool_class, Tool):
                                tool_instance = tool_class()
                                self.tool_registry.register_tool(tool_instance)
                                print(f"Registered custom tool: {tool_class.__name__}")
                        except (ImportError, AttributeError) as e:
                            print(f"Error loading custom tool {tool_name}: {e}")
                
                # Select system prompt
                base_prompt = TOOL_ENABLED_SYSTEM_PROMPT
                if config.tools.system_prompt == "math":
                    base_prompt = DEEPSEEK_MATH_SYSTEM_PROMPT
                
                # Create tool-enabled system prompt
                self.tool_system_prompt = self.tool_registry.get_system_prompt_with_tools(base_prompt)
                
                self.parse_tool_calls = parse_tool_calls
                self.insert_tool_responses = insert_tool_responses
                self.tools_available = True
                
                # Print startup message about tools
                print("Tool utilities initialized successfully")
                print(f"Available tools: {', '.join(self.tool_registry.tools.keys())}")
            except ImportError as e:
                print(f"Tool utilities not available, disabling tool use: {e}")
                self.tools_available = False
        else:
            print("Tools disabled in config or config not provided")

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        from concurrent.futures import ThreadPoolExecutor
        from typing import Dict, Any
        
        def process_item(args):
            i, data_item, already_print_data_sources = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            
            # Handle tool calls if enabled
            if self.tools_available and '<tool' in sequences_str:
                # Parse and execute tool calls
                tool_calls = self.parse_tool_calls(sequences_str)
                
                # Execute each tool call and collect responses
                tool_responses = []
                for tool_call in tool_calls:
                    response = self.tool_registry.execute_tool(tool_call)
                    tool_responses.append(response)
                
                # Insert tool responses into the text
                if tool_responses:
                    sequences_str = self.insert_tool_responses(sequences_str, tool_responses)
                    
                    # Store tool usage metrics in non_tensor_batch for later analysis
                    # This is informational only and not used in the current reward function
                    if 'tool_usage' not in data_item.non_tensor_batch:
                        data_item.non_tensor_batch['tool_usage'] = {}
                    
                    data_item.non_tensor_batch['tool_usage']['num_calls'] = len(tool_calls)
                    data_item.non_tensor_batch['tool_usage']['success_rate'] = sum(1 for r in tool_responses if r.success) / len(tool_responses) if tool_responses else 0

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            
            # For debug printing if needed
            # if data_source not in already_print_data_sources:
            #     already_print_data_sources[data_source] = 0
            # if already_print_data_sources[data_source] < self.num_examine:
            #     already_print_data_sources[data_source] += 1
            #     print(sequences_str)
                
            return i, score, valid_response_length

        # Process items in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=96) as executor:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(executor.map(process_item, args))

        # Fill reward tensor with results
        for i, score, valid_response_length in results:
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, config=config)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, config=config)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()

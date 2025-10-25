import yaml
import os
import time
import re
import asyncio
from http import HTTPStatus
from typing import AsyncGenerator, List, Literal, Optional, Union, Dict
import json

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
import logging
# lmdeploy
from lmdeploy.archs import get_task ## 判断多模态可以直接扔掉
from lmdeploy.messages import (GenerationConfig, PytorchEngineConfig,
                               TurbomindEngineConfig)
from lmdeploy.model import ChatTemplateConfig
# from lmdeploy.serve.async_engine import AsyncEngine
from transformers import AutoTokenizer
from lmdeploy.utils import get_logger
import sys
from async_engine import AsyncEngine
from protocol import (  # noqa: E501
    CompletionRequest as LmdeployCompletionRequest,CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse, ErrorResponse, ModelCard,
    ModelList, ModelPermission, UsageInfo, LogProbs)
from lmdeploy.tokenizer import DetokenizeState, Tokenizer
import copy

import sys



yaml_path='/lmdeploy_deploy/lmdeploy/lmdeploy_server_config.yaml'


class VariableInterface:
    backend: str = None
    async_engine = None
    session_id: int = 0
    api_keys: Optional[List[str]] = ['7f6cd4662cffb99bd51b61ed75b5e138']
    request_hosts = []
    rag_engine = None
    tokenizer = None

app = FastAPI()
get_bearer_token = HTTPBearer(auto_error=False)

def lmdeploy_engine(model_path, model_name, backend, backend_config_path, tp, log_level):
    if os.getenv('TM_LOG_LEVEL') is None:
        os.environ['TM_LOG_LEVEL'] = log_level
    logger = get_logger('lmdeploy', log_file='')
    logger.setLevel(log_level)
    ### 提取
    with open(backend_config_path, 'r') as file:
        config = yaml.safe_load(file)
    session_len = config.get('session_len', None)
    max_batch_size = config.get('max_batch_size', 128)
    # pipeline_type, pipeline_class = get_task(model_path)
    return AsyncEngine(
        model_path=model_path,
        model_name=model_name,
        backend=backend,
        backend_config=TurbomindEngineConfig(session_len=session_len, max_batch_size=max_batch_size, tp=tp, quant_policy=0) if backend == 'turbomind' else PytorchEngineConfig(session_len=session_len, max_batch_size=max_batch_size, tp=tp),
        chat_template_config=None,
        tp=tp)
    

async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    """Check if client provide valid api key.
    Adopted from https://github.com/lm-sys/FastChat/blob/v0.2.35/fastchat/serve/openai_api_server.py#L108-L127
    """  # noqa
    if VariableInterface.api_keys:
        if auth is None or (
                token := auth.credentials) not in VariableInterface.api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    'error': {
                        'message': 'Please request with valid api key!',
                        'type': 'invalid_request_error',
                        'param': None,
                        'code': 'invalid_api_key',
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None
    
def get_model_list():
    """Available models.

    If it is a slora serving. The model list would be [model_name,
    adapter_name1, adapter_name2, ...]
    """
    model_names = [VariableInterface.async_engine.model_name]
    cfg = VariableInterface.async_engine.backend_config
    model_names += getattr(cfg, 'adapters', None) or []
    return model_names

async def check_request(request) -> Optional[JSONResponse]:
    """Check if a request is valid."""
    if request.model in get_model_list():
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND, f'The model `{request.model}` does not exist.')
    return ret

def _create_completion_logprobs(tokenizer: Tokenizer,
                                token_ids: List[int] = None,
                                logprobs: List[Dict[int, float]] = None,
                                skip_special_tokens: bool = True,
                                offset: int = 0,
                                all_token_ids: List[int] = None,
                                state: DetokenizeState = None):
    """create openai LogProbs for completion.

    Args:
        tokenizer (Tokenizer): tokenizer.
        token_ids (List[int]): output token ids.
        logprobs (List[Dict[int, float]]): the top logprobs for each output
            position.
        skip_special_tokens (bool): Whether or not to remove special tokens
            in the decoding. Default to be True.
        offset (int): text offset.
        all_token_ids (int): the history output token ids.
        state (DetokenizeState): tokenizer decode state.
    """
    if logprobs is None or len(logprobs) == 0:
        return None, None, None, None

    if all_token_ids is None:
        all_token_ids = []
    if state is None:
        state = DetokenizeState()

    out_logprobs = LogProbs()
    out_logprobs.top_logprobs = []
    for token_id, tops in zip(token_ids, logprobs):
        out_logprobs.text_offset.append(offset)
        out_logprobs.token_logprobs.append(tops[token_id])

        res = {}
        out_state = None
        for top_id, prob in tops.items():
            response, _state = tokenizer.detokenize_incrementally(
                all_token_ids + [top_id],
                copy.deepcopy(state),
                skip_special_tokens=skip_special_tokens)
            res[response] = prob
            if top_id == token_id:
                out_state = _state
                offset += len(response)
                out_logprobs.tokens.append(response)

        out_logprobs.top_logprobs.append(res)
        state = out_state
        all_token_ids.append(token_id)

    return out_logprobs, offset, all_token_ids, state


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.get('/v1/models', dependencies=[Depends(check_api_key)])
def available_models():
    """Show available models."""
    model_cards = []
    for model_name in get_model_list():
        model_cards.append(
            ModelCard(id=model_name,
                      root=model_name,
                      permission=[ModelPermission()]))
    return ModelList(data=model_cards)
       
def create_error_response(status: HTTPStatus, message: str):
    """Create error response according to http status and message.

    Args:
        status (HTTPStatus): HTTP status codes and reason phrases
        message (str): error message
    """
    return JSONResponse(
        ErrorResponse(message=message,
                      type='invalid_request_error',
                      code=status.value).model_dump())   
    
@app.post('/v1/completions', dependencies=[Depends(check_api_key)])
async def completions_v1(raw_request: Request = None):
    request_json = await raw_request.json()
    
    if VariableInterface.backend in ['turbomind', 'pytorch']:
        VariableInterface.session_id += 1
        request = LmdeployCompletionRequest(**request_json)
        request.session_id = VariableInterface.session_id
        # error_check_ret = await check_request(request)
        # if error_check_ret is not None:
        #     return error_check_ret

        model_name = request.model
        adapter_name = None
        if model_name != VariableInterface.async_engine.model_name:
            adapter_name = model_name  # got a adapter name
        request_id = str(request.session_id)
        created_time = int(time.time())
        ### 测试一下这个stop有没有用
        # request.stop = None
        # if isinstance(request.prompt, list):
        #     request.prompt = [request.prompt]
        if isinstance(request.stop, str):
            request.stop = [request.stop]
            
        ###  定死重复生成
        request.repetition_penalty=1.0
        request.top_k = 5
        request.top_p = 0.9
        request.temperature = 0.000001
        
        gen_config = GenerationConfig(
            max_new_tokens=request.max_tokens if request.max_tokens else 512,
            logprobs=request.logprobs,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            ignore_eos=request.ignore_eos,
            stop_words=request.stop,
            skip_special_tokens=request.skip_special_tokens)
        
        generators = []
        for i in range(len(request.prompt)):
            result_generator = VariableInterface.async_engine.generate(
                request.prompt[i],
                request.session_id + i,
                gen_config=gen_config,
                stream_response=True,  # always use stream to enable batching
                sequence_start=True,
                sequence_end=True,
                do_preprocess=False,
                adapter_name=adapter_name)
            generators.append(result_generator)

        def create_stream_response_json(
            index: int,
            text: str,
            finish_reason: Optional[str] = None,
        ) -> str:
            choice_data = CompletionResponseStreamChoice(
                index=index,
                text=text,
                finish_reason=finish_reason,
            )
            response = CompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=model_name,
                choices=[choice_data],
            )
            response_json = response.model_dump_json()

            return response_json

        async def completion_stream_generator() -> AsyncGenerator[str, None]:
            # First chunk with role
            for generator in generators:
                for i in range(request.n):
                    choice_data = CompletionResponseStreamChoice(
                        index=i,
                        text='',
                        finish_reason=None,
                    )
                    chunk = CompletionStreamResponse(id=request_id,
                                                    choices=[choice_data],
                                                    model=model_name)
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f'data: {data}\n\n'

                async for res in generator:
                    response_json = create_stream_response_json(
                        index=0,
                        text=res.response,
                        finish_reason=res.finish_reason,
                    )
                    yield f'data: {response_json}\n\n'
            yield 'data: [DONE]\n\n'

        # Streaming response
        if request.stream:
            # flag = 1
            return StreamingResponse(completion_stream_generator(),
                                    media_type='text/event-stream')

        # Non-streaming response
        usage = UsageInfo()
        choices = []

        async def _inner_call(i, generator):
            final_res = None
            text = ''
            async for res in generator:
                if await raw_request.is_disconnected():
                    # Abort the request if the client disconnects.
                    await VariableInterface.async_engine.stop_session(
                        request.session_id)
                    return create_error_response(HTTPStatus.BAD_REQUEST,
                                                'Client disconnected')
                final_res = res
                text += res.response
            assert final_res is not None
            ### zxb::需要截断
            # if print_ctrl:
            #     logger = get_logger('lmdeploy')
            #     logger.info(f"===TEXT BEFORE SPLIT===\n{text}")
            #     text = re.split(r'\nHuman|\nAssistant|<\|endoftext\||<\|end', text)[0]
            #     if text.endswith('<') or text.endswith('\n'):
            #         text = text[:-1]
            #     elif text.endswith('<|') or text.endswith('\n\n'):
            #         text = text[:-2]
            #     elif text.endswith('\n<|') or text.endswith('\n\n\n'):
            #         text = text[:-3]
            #     else :
            #         pass
            #     logger.info(f"===TEXT AFTER SPLIT===\n{text}")
            
            ###
            choice_data = CompletionResponseChoice(
                index=0,
                text=text,
                finish_reason=final_res.finish_reason,
            )
            choices.append(choice_data)

            total_tokens = sum([
                final_res.history_token_len, final_res.input_token_len,
                final_res.generate_token_len
            ])
            usage.prompt_tokens += final_res.input_token_len
            usage.completion_tokens += final_res.generate_token_len
            usage.total_tokens += total_tokens

        await asyncio.gather(
            *[_inner_call(i, generators[i]) for i in range(len(generators))])

        response = CompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )
        total_info = {'prompt': request.prompt, 'response': response}
        print(total_info)

        return response
            # from control_repetiation.seq_rep_n_gram import calculate_seq_rep_all
            # flag = calculate_seq_rep_all(choices[-1].text)
            # print(flag)
        
        # total_info = {'prompt': request.prompt, 'response': response}
        # with open('/mnt/data/users/zhouxiabin/InitiAIInferenceSystem/log/0520_log.jsonl', 'a') as f:
        #     f.write(json.dumps(total_info, ensure_ascii=False) + '\n')
    
    
def serve(config_path: str = yaml_path):
    ## 读取配置文件
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    ### API 外部设置
    server_name = config['server'].get('name')
    server_port = config['server'].get('port')
    ### 模型设置
    model_path = config['model'].get('path')
    model_name = config['model'].get('name')
    ### 后端设置
    backend = config['backend'].get('type')
    backend_config_path = config['backend'].get('config')  # This should be instantiated based on your backend configuration classes
    tp = config['tensor_paral'].get('gpus', 1)
    ### PromptEngine
    chat_template_config = config['chat_template'].get('config')  # 一般用不到直接置为空就行
    log_level = config['logging'].get('level', 'INFO') 
    ### RAG Engine
    # is_rag_engine = config['rag'].get('is_rag_engine', True) 
    
    engine_mapping = {
        'turbomind': lambda: lmdeploy_engine(model_path, model_name, backend, backend_config_path, tp, log_level),
        'pytorch': lambda: lmdeploy_engine(model_path, model_name, backend, backend_config_path, tp, log_level),
    }
    if backend in engine_mapping:
        VariableInterface.async_engine = engine_mapping[backend]()
        VariableInterface.backend = backend
    VariableInterface.tokenizer = AutoTokenizer.from_pretrained(model_path)
    # if is_rag_engine:
    #     VariableInterface.rag_engine = FaissVectorDB()
    #     print('db loaded')
    
    ### API keys
    apikey = config['api_keys'].get('keys', None)
    if apikey:
        VariableInterface.api_keys += [apikey]
    
    # CORS setup 跨域设置
    cors_config = config.get('cors', {})
    if cors_config:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_config.get('allow_origins', ['*']),
            allow_credentials=cors_config.get('allow_credentials', True),
            allow_methods=cors_config.get('allow_methods', ['*']),
            allow_headers=cors_config.get('allow_headers', ['*']),
        )

    # SSL configuration
    ssl_config = config.get('ssl', {})
    if ssl_config.get('enabled', False):
        ssl_keyfile = os.environ.get('SSL_KEYFILE')
        ssl_certfile = os.environ.get('SSL_CERTFILE')
        uvicorn.run(app=app, host=server_name, port=server_port, log_level='info', ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile)
    else:
        uvicorn.run(app=app, host=server_name, port=server_port, log_level='info')

if __name__ == '__main__':
    import fire
    fire.Fire(serve)
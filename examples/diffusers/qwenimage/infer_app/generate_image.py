import asyncio
import pickle

import aiohttp


# 设置 Client 访问
def call_api_gen(url):
    async def _fn(samples, *args, **kwargs):
        data = {
            "prompts": samples,
            "num_inference_steps": 50,
            "true_cfg_scale": 4.0,
        }

        async with aiohttp.ClientSession() as sess:
            data_bytes = pickle.dumps(data)
            async with sess.get(url, data=data_bytes, timeout=12000) as response:
                result = bytearray()
                while not response.content.at_eof():
                    chunk = await response.content.read(1024)
                    result += chunk
                response_data = pickle.loads(result)
        return response_data

    return _fn


# 推理参数
prompt = "A beautiful sunset over the mountains"

# 设置两个服务端 URL
url1 = "http://127.0.0.1:5000/qwenimage-api"
url2 = "http://127.0.0.1:5001/qwenimage-api"

# 创建两个管道
pipe1 = call_api_gen(url1)
pipe2 = call_api_gen(url2)


# 并行发送请求
async def run_both():
    # 同时执行两个异步任务
    result1, result2 = await asyncio.gather(pipe1(prompt), pipe2(prompt))
    return result1, result2


# 执行并获取结果
results = asyncio.run(run_both())

# 保存图像, 所有图像都是相同的, 保存第一个即可
results[0].save("generated_image.png")
# image.show()

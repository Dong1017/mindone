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


port = 5000
worker_num = 4

# 动态生成 URL 和管道
urls = [
    f"http://127.0.0.1:{port + i}/qwenimage-api"
    for i in range(worker_num)
]
pipes = [call_api_gen(url) for url in urls]

# ✅ 修改：把 prompt 作为参数传入
async def run_all(pipes, prompt):
    results = await asyncio.gather(
        *[pipe(prompt) for pipe in pipes]
    )
    return results

# 推理参数
# prompt = "A beautiful sunset over the mountains"
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'''

# 执行并获取结果
results = asyncio.run(run_all(pipes, prompt))

# 保存图像, 所有图像都是相同的, 保存第一个即可
results[0].save("generated_image.png")
# image.show()

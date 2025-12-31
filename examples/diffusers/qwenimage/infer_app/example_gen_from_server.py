import asyncio


# calling function
def call_api_gen(url):
    async def _fn(samples, *args, **kwargs):
        # ==================== We comment the follow code, because security risks.  ======================
        # ===========           You need to manually decomment it before running.            =============
        # ========                              The first place.                                ==========
        # ================================================================================================
        #
        # import aiohttp
        # import pickle
        # async with aiohttp.ClientSession() as sess:
        #     data = {
        #         "prompts": samples,
        #         "num_inference_steps": 50,
        #         "true_cfg_scale": 4.0,
        #     }
        #     data_bytes = pickle.dumps(data)
        #     timeout = 15000  # bigger than the estimated inference time cost (second)
        #     async with sess.get(url, data=data_bytes, timeout=timeout) as response:
        #         result = bytearray()
        #         while not response.content.at_eof():
        #             chunk = await response.content.read(1024)
        #             result += chunk
        #         response_data = pickle.loads(result)
        # return response_data
        #
        # ================================================================================================

        raise NotImplementedError(
            "There are some security risks from pickle here. \n"
            "You need to confirm it and manually decomment the code above before running them."
        )

    return _fn


# default parameters
port = 5000
worker_num = 4

# get pipes for different ports
urls = [f"http://127.0.0.1:{port + i}/qwenimage-api" for i in range(worker_num)]
pipes = [call_api_gen(url) for url in urls]


# function for passing inference requests
async def run_all(pipes, prompt):
    results = await asyncio.gather(*[pipe(prompt) for pipe in pipes])
    return results


# inference parameters
prompt = (
    'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light '
    'beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the '
    'poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'
)

# call and get results from server
results = asyncio.run(run_all(pipes, prompt))

# save image
results[0].save("generated_image.png")

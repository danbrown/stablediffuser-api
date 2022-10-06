from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from pydantic import BaseModel
import base64
from io import BytesIO
import nest_asyncio
import uvicorn
import json
import subprocess

from source.classes.manager import Manager
from source.classes.cleaner import Cleaner
from source.generate import GenerationRequest, generate_to_base64

global pipetype
global pipe
global last_model

if __name__ == "__main__":
  # start pipe and get settings
  Cleaner.clean_env()
  manager = Manager()
  pipe, pipetype, last_model= manager.eval_settings()

  # Create fast api app
  app = FastAPI()

  app.add_middleware(
      CORSMiddleware,
      allow_origins=['*'],
      allow_credentials=True,
      allow_methods=['*'],
      allow_headers=['*'],
  )

  # Api routes
  @app.get('/')
  async def root(response: Response):
    # dependency free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0: # success state on shell command
        gpu = True
    response = {"state": "healthy", "gpu": gpu}
    return Response(content=json.dumps(response), media_type="application/json")

  @app.post('/')
  async def generateRoot(generationRequest: GenerationRequest, response: Response):
    requestParameters = generationRequest.dict()
    # requestParameters["num_iters"] = 1 # force num_iters to 1

    # generate image and update types
    global pipetype
    global pipe
    global last_model
    images, pipe, pipetype, last_model = generate_to_base64(pipe=pipe, pipetype=pipetype, last_model=last_model, parameters=requestParameters)


    # IMAGE RETURN
    # response.headers["X-seed"] = str(images[0]["seed"])
    # return StreamingResponse(BytesIO(images[0]["image"]), media_type="image/png")

    # HTML RETURN
    # html = """<!DOCTYPE html><html><head><title>Generated Images</title></head><body>"""
    # for image in images:
    #   html += f"""<img src="{image["image"]}">"""
    # html += """</body></html>"""
    # return html
    # return HTMLResponse(content=html, status_code=200)

    # JSON RETURN

    # empty dict
    result = { "parameters": requestParameters }

    result["images"] = []
    result["images"].extend(images)
    result = json.dumps(result)

    return Response(content=result, media_type="application/json")
 
  # starting app 
  nest_asyncio.apply()
  uvicorn.run(app, port=7860, host='0.0.0.0')
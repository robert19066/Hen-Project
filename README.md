
![if you see this this means you have no wifi or smth went wrong and you can't see the gorgeous logo poor you](https://i.postimg.cc/28XgstvF/Gemini-Generated-Image-fenw7rfenw7rfenw.png)

# Hen AI[Documentation updated to V4]

Hen is a fine-tunable LLM designed for interactive chat and text generation. Supports images and any file type you can think of!

# Features

##### _(note the features tab only updates when a new feature is added)_

**Updated since version: V4**

- A web UI for chats with model selection.
- Added training script and a variety of models to choose.
- GPU compatability tester.
- CLI Trainer and Chat.
- Comand-line trainer for in-app use(like in other apps)
- Transforming your device into a Hen Endpoint using Hen Endpot

# Instalation:

Firstly make an python virtual env(conda has not been tested, venv is recomanded) and run:
`pip install requirements.txt`
to install all dependencies. Then, from the env, you can run the script you wish!
Congrats, Hen is succesfully installed!

# How2use:

## List of scripts

And what they do(all are in the folder Scripts):

- `gpu_test.py` - Tests your gpu compatability(cpu is not supported)
- `hen_chat.py` - The CLI chat
- `hen_trainer.py` - The CLI trainer
- `hen_endpot.py` - The Hen ENDPOT server,see lower for details.
- `hen_cmd.py` - Command-based manager, in case you need to use Hen in other projects!
- (AT THE ROOT FOLDER, NOT AT SCRIPTS)`app.py` - The backend for the Chat Web UI.

## Endpot(**NEW**):
![ENDPOTTT](https://i.postimg.cc/5tQWt66X/Gemini-Generated-Image-7qmact7qmact7qma.png)
I am exited to show you this new feature i worked one, its massive because now i'm focusing on making Hen implementable in actual projects. This is no longer a experiment, but a whole blown product! -  you can turn your **own** pc into a Hen Endpoint! You can set adress, port, max tokens, temperature and even an API KEY!
The script `hen_endpot.py` handles it all, easely, with a friendly CLI!

### ENDPOT Usage
- #### List Models

```bash
curl -X POST http://localhost:5000/api/hen \
 -H "Content-Type: application/json" \
 -d '{"action": "list"}'
``````

#### Response:
``````json
{
  "success": true,
  "action": "list",
  "count": 2,
  "models": [
    {
      "index": 1,
      "name": "hen_o2_mini_20250122_143022",
      "tier": "MINI",
      "created_at": "2025-01-22 14:30:22"
    }
  ],
  "sort_order": "new_to_old"
}
``````


- #### Run Inference by Index
```bash
curl -X POST http://localhost:5000/api/hen \
 -H "Content-Type: application/json" \
 -d '{
"action": "run",
"modelIndex": 1,
"modelIndexSort": "new_to_old",
"container": "Explain quantum computing"
}'
```


- #### Run Inference by Name
```bash
curl -X POST http://localhost:5000/api/hen \
  -H "Content-Type: application/json" \
  -d '{
    "action": "run",
    "model": "my_code_model",
    "container": "Write a Python function to reverse a string"
  }'

```


- #### With API Key (if enabled)
```bash
curl -X POST http://localhost:5000/api/hen \
 -H "Content-Type: application/json" \
 -H "X-API-Key: your-secret-key" \
 -d '{"action": "list"}'
```



- #### Check Status:
```bash
curl -X POST http://localhost:5000/api/hen \
  -H "Content-Type: application/json" \
  -d '{"action": "status"}'
```

# LICENCE

The project is licensed under the _Apache Licence 2.0_.

### To note:

The documentation can sometimes be older than the version,thats why you can see from what version it was updated(to avoid confusion)

## Message for who wants to use Hen in their projects

Please give credit, taking an other person's work and not giving them credit is **STEALING!** Stealing is bad, so give credit.
It doesn't hurt!

# Awarded contribuitors 🌟

none....YET! Shall you be the first one?(yes the fist ever contribuitor is awarded)

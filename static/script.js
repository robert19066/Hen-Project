document.addEventListener('DOMContentLoaded', () => {

    scanModels();

});



let selectedImage = null;



// #########################################################################################################################
// HEN WEBUI JS V1                                                                                                        //
// Major Updates:                                                                                                         //
// ITS THE FIRST VERSION                                                                                                  //
// ##########################################################################################################################



async function scanModels() {

    const select = document.getElementById('model-select');

    try {

        const response = await fetch('/scan');

        const models = await response.json();

        

        select.innerHTML = '<option value="base">0. Use Base Model (No Fine-tuning)</option>';

        models.forEach((m, index) => {

            const option = document.createElement('option');

            option.value = m.id;

            option.text = `${index + 1}. ${m.name} (${m.tier})`;

            select.appendChild(option);

        });

    } catch (e) {

        select.innerHTML = '<option>Error scanning models</option>';

    }

}


async function loadSelectedModel() {
    const select = document.getElementById('model-select');
    const status = document.getElementById('model-status');
    const btn = document.getElementById('load-btn');
    
    // Show loading state
    status.innerHTML = '<div class="spinner"></div> Hatching the Hen...';
    status.className = "status-indicator loading";
    btn.disabled = true;

    try {
        const response = await fetch('/load_model', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ path: select.value })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            status.innerHTML = `✅ Active: ${data.loaded}`;
            status.className = "status-indicator";
            status.style.color = "#4caf50";
        } else {
            throw new Error(data.error);
        }
    } catch (e) {
        status.innerHTML = "❌ Load failed: " + e.message;
        status.className = "status-indicator";
        status.style.color = "red";
    } finally {
        btn.disabled = false;
    }
}



// --- Chat Logic ---



function handleImageUpload(input) {

    if (input.files && input.files[0]) {

        selectedImage = input.files[0];

        const reader = new FileReader();

        reader.onload = function(e) {

            document.getElementById('preview-img').src = e.target.result;

            document.getElementById('image-preview').style.display = 'block';

        }

        reader.readAsDataURL(input.files[0]);

    }

}



function clearImage() {

    selectedImage = null;

    document.getElementById('image-upload').value = "";

    document.getElementById('image-preview').style.display = 'none';

}



async function sendMessage() {

    const input = document.getElementById('user-input');

    const chatHistory = document.getElementById('chat-history');

    const roMode = document.getElementById('ro-mode').checked;

    

    const text = input.value.trim();

    if (!text && !selectedImage) return;



    // Add User Message to UI

    appendMessage('user', text, selectedImage);

    input.value = '';

    

    // Prepare Data

    const formData = new FormData();

    formData.append('text', text);

    formData.append('romanian', roMode);

    if (selectedImage) {

        formData.append('image', selectedImage);

        clearImage();

    }



    // Add Bot Placeholder

    const botBubble = appendMessage('bot', 'Thinking...');



    try {

        const response = await fetch('/chat', {

            method: 'POST',

            body: formData

        });



        const reader = response.body.getReader();

        const decoder = new TextDecoder();

        let resultText = "";



        while (true) {

            const { done, value } = await reader.read();

            if (done) break;

            const chunk = decoder.decode(value);

            resultText += chunk;

            botBubble.textContent = resultText; // Simple streaming update

            chatHistory.scrollTop = chatHistory.scrollHeight;

        }

    } catch (e) {

        botBubble.textContent = "Error: Could not connect to Hen.";

        botBubble.style.color = "red";

    }

}



function appendMessage(role, text, imageFile=null) {

    const history = document.getElementById('chat-history');

    const msgDiv = document.createElement('div');

    msgDiv.className = `message ${role}`;

    

    let content = "";

    if (imageFile) content += `[Image Uploaded]<br>`;

    content += text.replace(/\n/g, '<br>');



    const bubble = document.createElement('div');

    bubble.className = 'bubble';

    bubble.innerHTML = content;

    

    msgDiv.appendChild(bubble);

    history.appendChild(msgDiv);

    history.scrollTop = history.scrollHeight;

    

    return bubble; // Return bubble for streaming updates

}

let net = null;
let processedValue = null;
let imageDataArray = null;

function showFiles() {
    // An empty img element
    let demoImage = document.getElementById('idImage');
    // read the file from the user
    let file = document.querySelector('input[type=file]').files[0];
    const reader = new FileReader();
    reader.onload = async function (event) {
        demoImage.src = reader.result;
        // Convert the image file to an array
        imageDataArray = await imageFileToArray(file);
        // Make the AJAX request with the image data array
        sendImageData(imageDataArray);
    }
    reader.readAsDataURL(file);
}

function sendImageData(imageDataArray) {
    // Get the CSRF token from the cookie
    const csrfToken = getCookie('csrftoken');

    // Make the AJAX request with the image data array
    $.ajax({
        type: 'POST',
        url: '/process_variable/',
        data: JSON.stringify({
            'variable_name': Array.from(imageDataArray)
        }),
        contentType: 'application/json', // Set content type to JSON
        beforeSend: function(xhr) {
            xhr.setRequestHeader("X-CSRFToken", csrfToken); // Set the CSRF token in the request headers
        },
        success: function(response) {
            // Handle the response from the Django view
            console.log('Processed value:', response.processed_value);
            processedValue = response.processed_value;
            
            updateDisplay(processedValue);
        },
        error: function(xhr, status, error) {
            console.error('AJAX error:', error);
        }
    });
}
// Function to update the display with the processed value
function updateDisplay(value) {
    const displayElement = document.getElementById('displayString');
    displayElement.textContent = value;
}


// Function to convert image file to an array
async function imageFileToArray(file) {
    return new Promise((resolve, reject) => {
        // Create a new FileReader
        let reader = new FileReader();

        // Set up onload event for the reader
        reader.onload = function(event) {
            // Create a new image element
            let img = new Image();

            // Set up onload event for the image
            img.onload = function() {
                // Create a canvas element
                let canvas = document.createElement('canvas');

                // Set the canvas dimensions to match the image
                canvas.width = img.width;
                canvas.height = img.height;

                // Get the 2D context of the canvas
                let ctx = canvas.getContext('2d');

                // Draw the image onto the canvas
                ctx.drawImage(img, 0, 0);

                // Get the pixel data from the canvas
                let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

                // Resolve the promise with the pixel data
                resolve(imageData.data);
            };

            // Set the source of the image to the loaded file data
            img.src = event.target.result;
        };

        // Read the file as a data URL
        reader.readAsDataURL(file);
    });
}



app();

// Function to get the CSRF token from the cookie
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

showFiles();
app();
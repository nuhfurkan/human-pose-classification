const constraints = {
    audio: true,
    video: {height: 300, width:300}
}

async function stream(myid) {
    const video = document.getElementById(myid);
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints)
      video.srcObject = mediaStream;
      localStream = mediaStream;
    } catch (e) {
      console.error(e)
    }
    video.onloadedmetadata = async function(event) {
      try {
        await video.play();
      } catch (e) {
        console.error(e)
      }
    }
}

function snap() {
   //context.fillRect(0,0,myWidth,myHeight);
   //context.drawImage(video,0,0,myWidth,myHeight);
   let canvas = document.getElementById('screenshot-canvas');
   let video = document.getElementById('video-elem');
   canvas.width = 800;
   canvas.height = 600;
   
   let ctx = canvas.getContext('2d');
   ctx.drawImage( video, 0, 0, canvas.width, canvas.height );
   
   const fileInput = document.querySelector('input[type="file"]');
   let image = canvas.toDataURL('image/png');

   canvas.toBlob(function(blob) {
    // Use the Blob object
    // e.g., send it to a server or process it further
    
    // Example: Create a URL for the Blob and display it in an image element
    var blobURL = URL.createObjectURL(blob);
    var formData = new FormData();

    formData.append("images", blob, "imagename");  
  
    // Send the FormData object via AJAX
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/upload', true);
    xhr.onload = function() {
      if (xhr.status === 200) {
        // Request successful
        var response = xhr.responseText;

        console.log('Images uploaded successfully');
        document.getElementById("feedback").textContent = response
        console.log(response)
      } else {
        // Error handling
        console.error('Error uploading images:', xhr.status);
      }
    };
    xhr.send(formData);
  
  }, 'image/jpeg', 0.8);

  // Create a new FormData object



   console.log("heyyo");
}

function startVideo() {
    stream("video-elem")
}

var video = document.querySelector("#video-elem");

if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
      video.srcObject = stream;
    })
    .catch(function (err0r) {
      console.log("Something went wrong!");
    });
}

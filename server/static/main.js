let form = document.forms.namedItem("fileinfo");
let lastImages = [];

form.addEventListener('submit', function (ev) {
  ev.preventDefault();

  document.getElementById("loader").style.visibility = "visible";
  let formdata = new FormData(form);

  // Get all the uploaded files
  const uploadedFiles = form.elements["file"].files;

  // Append each file to the formdata separately
  for (let i = 0; i < uploadedFiles.length; i++) {
    formdata.append("file", uploadedFiles[i]);
  }

  let requestOptions = {
    method: 'POST',
    body: formdata,
  };

  let loc = window.location;

  fetch(`${loc.protocol}//${loc.hostname}:${loc.port}/predict`, requestOptions)
    .then(response => response.json())
    .then(results => {
      console.log(results);
      DisplayResult(results, uploadedFiles);
      document.getElementById("loader").style.visibility = "hidden";
    });
});

// On Images Uploaded
form.addEventListener('change', function (ev) {
  lastImages = [];

  // Get all the uploaded files
  const uploadedFiles = form.elements["file"].files;

  for (let i = 0; i < uploadedFiles.length; i++) {
    lastImages.push(URL.createObjectURL(uploadedFiles[i]));
  }

  // Display all the uploaded images
  const imageContainer = document.getElementById("uploadedImages");
  imageContainer.innerHTML = ""; // Clear previous images

  for (let i = 0; i < lastImages.length; i++) {
    let imgElement = document.createElement("img");
    imgElement.src = lastImages[i];
    imgElement.className = "uploaded-img";
    imageContainer.appendChild(imgElement);
  }
});

function DisplayResult(results, uploadedFiles) {
  let resultDiv = document.getElementById("results");
  resultDiv.innerHTML = ""; // Clear previous results

  for (let i = 0; i < uploadedFiles.length; i++) {
    let divContainer = document.createElement("div");
    divContainer.className = "result";

    let divImage = document.createElement("div");
    divImage.className = "image";
    divImage.style.backgroundImage = "url('" + lastImages[i] + "')";

    let divText = document.createElement("div");
    divText.appendChild(document.createTextNode(results[i])); // Use the correct result for each image
    divText.className = "text";

    divContainer.appendChild(divImage);
    divContainer.appendChild(divText);
    resultDiv.appendChild(divContainer); // Use appendChild instead of prepend to maintain order
  }
}

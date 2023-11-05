
const dropZone = document.querySelector('.drop-zone');
var fileInput = document.querySelector('#file-input');
const browseBtn = document.querySelector('.browseBtn');
const video = document.querySelector(".video-container");
const analyzeBtn=document.querySelector("#analyzeBtn");

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  if (!dropZone.classList.contains('dragged'))
    dropZone.classList.add('dragged');
});

document.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragged');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragged');
  const files = e.dataTransfer.files;
  // console.table(files);
  if (files.length) {
    fileInput.files = files;
    playSelectedFile();
  }
});

const downloadToFile = (content, filename, contentType) => {
  const a = document.createElement('a');
  const file = new Blob([content], {type: contentType});
  
  a.href= URL.createObjectURL(file);
  a.download = filename;
  a.click();

	URL.revokeObjectURL(a.href);
};

browseBtn.addEventListener('click', () => {
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  playSelectedFile();
});

function UploadFile() {
  let formData = new FormData(); 
  formData.append("file", fileInput.files[0]);
  console.log(fileInput.files[0]);
  downloadToFile(fileInput.files[0], 'video_selected', 'video');
  location.href = 'result.html';
}
analyzeBtn.addEventListener('click', () =>{
  UploadFile();
}); 
const playSelectedFile = function () {
  if (fileInput.files.length > 1) {
    fileInput.value = "";
    return;
  }
  const file = fileInput.files[0];
  const type = file.type;
  const videoNode = document.querySelector('video');
  const canPlay = videoNode.canPlayType(type);
  if (canPlay === ''){
    alert("Cannot play the media");
    window.location="http://127.0.0.1:5000/";
  } //canPlay = 'no';
  const message = 'Can play type "' + type + '": ' + canPlay;
  const isError = canPlay === 'no';
  console.log(canPlay);
  if (isError) {
    alert(message);
    return;
  }

  //video.style.display = 'block';
  //dropZone.style.display = 'none';
  const fileURL = URL.createObjectURL(file);
  videoNode.src = fileURL;
}
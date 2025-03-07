function showInfo() {
    const infoBox = document.getElementById("info-box");
    if (infoBox.style.display === "none" || infoBox.style.display === "") {
        infoBox.style.display = "block";  // Show the info box
    } else {
        infoBox.style.display = "none";  // Hide the info box
    }
}


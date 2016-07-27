'use strict';
function Modal() {
}

/** Init Modal functions **/
Modal.prototype.init = function () {
  var self = this;

  // init handlers
  self.initHandlers();
};

/** Initialization of handlers on top menu **/
Modal.prototype.initHandlers = function () {
  var self = this;
  var emailBtnsCollection = document.getElementsByClassName('email-btn');
  var overlay = document.getElementById("modal-overlay");
  var contactForm = document.getElementById('contact-form');

  // handle click on send email button and show email form
  for (var i = 0; i < emailBtnsCollection.length; i++) {
    emailBtnsCollection[i].addEventListener("click", function (event) {
      event.preventDefault();
      self.showOverlay(overlay);
      self.showContactForm(contactForm);
    });
  }

  // hide overlay if user clicked on it
  overlay.addEventListener("click", function (e) {
    e.preventDefault();
    self.hideOverlay(overlay);
    self.hideContactForm(contactForm);
  });

  // function to send email and show success window
  document.getElementById('contact-form').addEventListener('submit', function (event) {
    event.preventDefault();
    self.submitEmail();
    self.hideOverlay(overlay);
    self.hideContactForm(contactForm);
  });

  // hide success modal
  document.getElementById('success-send-close-btn').addEventListener("click", function (e) {
    e.preventDefault();
    self.hideOverlay(overlay);
    self.hideContactForm(contactForm);
  });
};

/** Open modal overlay **/
Modal.prototype.showOverlay = function (overlay) {
  overlay.style.display = "block";
  setTimeout(function () {
    overlay.classList.add('visible');
  }, 100)
};

/** Close modal overlay **/
Modal.prototype.hideOverlay = function (overlay) {
  overlay.classList.remove('visible');
  setTimeout(function () {
    overlay.removeAttribute("style");
  }, 400);
};

/** Show modal contact form with animation **/
Modal.prototype.showContactForm = function (contactForm) {
  contactForm.classList.remove('hidden'); // make contact form being displayed, but still invisible

  // make it visible with certain animations
  setTimeout(function () {
    contactForm.style.opacity = 1;
    contactForm.style.top = '22%';
  }, 100)
};

/** Hide modal contact form with animation **/
Modal.prototype.hideContactForm = function (contactForm) {
  contactForm.removeAttribute('style');

  // make it visible with certain animations
  setTimeout(function () {
    contactForm.classList.add('hidden');
  }, 400)
};

/** Send email message when submit clicked **/
Modal.prototype.submitEmail = function () {
  var self = this;
  var elements = document.getElementsByClassName("form-val");
  var formData = {};
  var xmlHttp = new XMLHttpRequest();

  for (var i = 0; i < elements.length; i++) {
    formData[elements[i].name] = elements[i].value;
  }

  xmlHttp.onreadystatechange = function () {
    if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
      self.showSuccessText();
    }
  };

  xmlHttp.open("POST", "https://formspree.io/d.bobrenko@gmail.com");
  xmlHttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  xmlHttp.send(JSON.stringify(formData));
};

/** Show success window if email sent was successful **/
Modal.prototype.showSuccessText = function (){
  var self = this;

  // hide form elements
  var elements = document.getElementsByClassName("form-val");
  for (var i = 0; i < elements.length; i++) {elements[i].style.display = 'none'};
  document.querySelector('#contact-form .send-btn').style.display = 'none';

  // show success fields
  document.querySelector('.success-send-msg').style.display = 'block';
  document.querySelector('.success-send-close-btn').style.display = 'block';

  // show modal window
  self.showOverlay(document.getElementById("modal-overlay"));
  self.showContactForm(document.getElementById('contact-form'));
};
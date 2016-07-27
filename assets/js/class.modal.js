'use strict';
function Modal(){
}

/** Init Modal functions **/
Modal.prototype.init = function (){
  var self = this;

  // init handlers
  self.initHandlers();
};

/** Initialization of handlers on top menu **/
Modal.prototype.initHandlers = function (){
  var self = this;
  var emailBtnsCollection = document.getElementsByClassName('email-btn');
  var overlay = document.getElementById("modal-overlay");
  var contactForm = document.getElementById('contact-form');

  // handle click on send email button and show email form
  for(var i = 0; i < emailBtnsCollection.length; i++){
    emailBtnsCollection[i].addEventListener("click", function (event){
      event.preventDefault();
      self.showOverlay(overlay);
      self.showContactForm(contactForm);
    });
  }

  // hide overlay if user clicked on it
  overlay.addEventListener("click", function (e){
    e.preventDefault();
    self.hideOverlay(overlay);
    self.hideContactForm(contactForm);
  })
};

/** Open modal overlay **/
Modal.prototype.showOverlay = function (overlay){
  overlay.style.display = "block";
  setTimeout(function (){
    overlay.classList.add('visible');
  }, 100)
};

/** Close modal overlay **/
  Modal.prototype.hideOverlay = function (overlay){
  overlay.classList.remove('visible');
  setTimeout(function (){
    overlay.removeAttribute("style");
  }, 400);
};

/** Show modal contact form with animation **/
Modal.prototype.showContactForm = function (contactForm){
  contactForm.classList.remove('hidden'); // make contact form being displayed, but still invisible

  // make it visible with certain animations
  setTimeout(function (){
    contactForm.style.opacity = 1;
    contactForm.style.top = '22%';
  }, 100)
};

/** Hide modal contact form with animation **/
Modal.prototype.hideContactForm = function (contactForm){
  contactForm.removeAttribute('style');

  // make it visible with certain animations
  setTimeout(function (){
    contactForm.classList.add('hidden');
  }, 400)
};
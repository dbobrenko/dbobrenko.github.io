'use strict';
function Navigation(){
}

/** Init navigation functions **/
Navigation.prototype.init = function (){
  var self = this;

  // init handlers
  self.initHandlers();
};

/** Initialization of handlers on top menu **/
Navigation.prototype.initHandlers = function (){
  var self = this;

  // define elements
  var topMenu = document.getElementById("top-nav");
  var topMenuToggle = document.getElementById("navbar-btn");

  topMenuToggle.addEventListener('click', function (event){
    event.preventDefault();
    if(topMenu.getAttribute('data-state') === 'opened') {
      self.hideTopMenu(topMenu);
    } else {
      self.showTopMenu(topMenu);
    } // end if statement
  }); // end addEventListener

  // close menu if user clicked on body
  document.addEventListener('click', function (event) {
    if(topMenu.getAttribute('data-state') === 'opened') {
      self.hideTopMenu(topMenu);
    }
  });
};

/** Drop-down top menu **/
Navigation.prototype.showTopMenu = function (topMenu){
  var height;

  // calculate height
  topMenu.style.height = 'auto';
  height = topMenu.offsetHeight;
  topMenu.style.height = '';

  // drop-down menu with animation
  setTimeout(function (){
    topMenu.style.height = height + 'px';
    topMenu.setAttribute('data-state', 'opened');
  }, 50);
};

/** Drop-up top menu **/
Navigation.prototype.hideTopMenu = function (topMenu){
  topMenu.removeAttribute('data-state');
  topMenu.removeAttribute('style');
};
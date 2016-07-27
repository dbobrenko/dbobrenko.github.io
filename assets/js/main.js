'use strict';
var navigation = new Navigation();
var modal = new Modal();

/** MAIN PAGE **/
var pageController = {

  /** Init site functions **/
  init: function (){
    navigation.init(); // initialization of main (top) menu
    modal.init(); // initialization of modal form controller
  }
};

document.addEventListener("DOMContentLoaded", pageController.init());
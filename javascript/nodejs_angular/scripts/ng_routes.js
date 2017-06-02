angular.module('app', ['ngRoute'])
  .config(['$routeProvider', function ($routeProvider) {
    console.log('here');
    $routeProvider
      .when('/', {
        templateUrl: 'todos.html',
        controller: 'TodoController'
      });
  }]);

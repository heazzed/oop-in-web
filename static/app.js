$(document).ready(function(){

  let token = localStorage.getItem("token");

  if (token !== null) {
    $("#login-form").hide();
  }else{
    $("#login-form").show();
  }

  $("#login-button").on("click", function (event) {
    event.preventDefault();
    $.ajax({
        type: 'POST',
        url: "/login",
    }).done(function (data) {
      token = data["token"];
      localStorage.setItem("token", token);
      $("#login-form").hide();
    }).fail(function () {
      console.log("Failed to login!");
    });
  });

  $("#create-button").on("click", function (e) {
    e.preventDefault();

    let seniority = $("#seniority").val();
    let home = $("#home").val();
    let time = $("#time").val();
	let age = $("#age").val();
    let marital = $("#marital").val();
	let records = $("#records").val();
    let job = $("#job").val();
	let expenses = $("#expenses").val();
    let income = $("#income").val();
	let assets = $("#assets").val();
    let debt = $("#debt").val();
	let amount = $("#amount").val();
    let price = $("#price").val();

    $.ajax({
        type: 'POST',
        url: "/clients",
        headers: {
            "Authorization":"Token: " + token
        },
        data: {"seniority": seniority, "home": home, "time": time, "age": age, "marital": marital, "records": records, "job": job, "expenses": expenses, "income": income, "assets": assets, "debt": debt, "amount": amount, "price": price},
        statusCode: {
          401: function() {
            alert('Authorize First!');
          }
        }
    }).done(function () {
      location.reload();
    }).fail(function () {
      console.log("Failed to create!");
    })
  });

  $(".delete-button").on("click", function (e) {
    let r = confirm("Sure you want to delete this sample?");
    let clientId = $(this).data("id");

    if (r === true) {
      $.ajax({
          type: 'DELETE',
          url: "/clients/"+clientId,
          headers: {
              "Authorization":"Token: " + token
          },
          statusCode: {
            401: function() {
              alert('Authorize First!');
            }
          }
      }).done(function () {
        location.reload();
      }).fail(function () {
        console.log("Failed to delete!" );
      });
    } else {
      console.log("You pressed Cancel!");
    }
  })
});
import 'package:dream11/login.dart';
import 'package:flutter/material.dart';

class SignUp extends StatefulWidget {
  const SignUp({super.key});

  @override
  State<SignUp> createState() => _SignUpState();
}

class _SignUpState extends State<SignUp> {
  TextEditingController _controller = TextEditingController();
  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          Expanded(
            child: Container(
              color: Colors.white,
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      "Sign Up",
                      style: TextStyle(
                        fontSize: 50,
                        fontWeight: FontWeight.w700,
                        color: Colors.grey[850],
                      ),
                    ),
                    SizedBox(height: 16),
                    Container(
                      width: 342,
                      height: 42,
                      color: Colors.grey[300],
                      child: Padding(
                        padding: EdgeInsets.only(left: 8),
                        child: TextField(
                          controller: _controller,
                          decoration: InputDecoration(
                            labelText: "Enter you email",
                            border: InputBorder.none,
                          ),
                        ),
                      ),
                    ),
                    SizedBox(height: 16),
                    Container(
                      width: 342,
                      height: 42,
                      color: Colors.grey[300],
                      child: Padding(
                        padding: EdgeInsets.only(left: 8),
                        child: TextField(
                          controller: _controller,
                          decoration: InputDecoration(
                            labelText: "Enter a password",
                            border: InputBorder.none,
                          ),
                        ),
                      ),
                    ),
                    SizedBox(height: 16),
                    Container(
                      width: 342,
                      height: 42,
                      color: Colors.grey[300],
                      child: Padding(
                        padding: EdgeInsets.only(left: 8),
                        child: TextField(
                          controller: _controller,
                          decoration: InputDecoration(
                            labelText: "Confirm password",
                            border: InputBorder.none,
                          ),
                        ),
                      ),
                    ),
                    SizedBox(height: 30),
                    GestureDetector(
                      onTap: (){
                        Navigator.push(
                          context,
                          MaterialPageRoute(builder: (context)=>SignUp())
                        );
                      },
                      child: Container(
                        padding: EdgeInsets.symmetric(horizontal: 15,vertical: 8),
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(50),
                          color: Colors.red[700]
                        ),
                        child: Text(
                          "Sign Up",
                          style: TextStyle(
                            fontWeight: FontWeight.w400,
                            fontSize: 18,
                            color: Colors.white
                          ),
                        ),
                      ),
                    )
                  ],
                ),
              ),
              )
          ),
          Expanded(
            child: Container(
              color: Colors.red[700],
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(
                      "Welcome Back",
                      style: TextStyle(
                        fontSize: 50,
                        fontWeight: FontWeight.w700,
                        color: Colors.white
                      ),
                    ),
                    Text(
                      "Already have an account?",
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w400,
                        color: Colors.white
                      ),
                    ),
                    SizedBox(height: 30),
                    GestureDetector(
                      onTap: (){
                        Navigator.push(
                          context,
                          MaterialPageRoute(builder: (context)=>LoginPage())
                        );
                      },
                      child: Container(
                        padding: EdgeInsets.symmetric(horizontal: 15,vertical: 8),
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(50),
                          border: Border.all(
                            color: Colors.white,
                            width: 1
                          )
                        ),
                        child: Text(
                          "Sign In",
                          style: TextStyle(
                            fontWeight: FontWeight.w400,
                            fontSize: 18,
                            color: Colors.white
                          ),
                        ),
                      ),
                    )
                  ],
                ),
              ),
              )
          ),
        ],
      ),
    );
  }
}
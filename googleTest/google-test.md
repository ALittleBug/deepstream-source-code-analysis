## We can check the unit test to understand the function of C++

* [offical doc](https://github.com/google/googletest/blob/master/googletest/docs/primer.md)
* [boqin youtube video](https://www.youtube.com/watch?v=iyDAFpcEy4c&list=PL5jc9xFGsL8GyES7nh-1yqljjdTvIFSsh&index=5)
  * Test fixture
    ```
    To create a fixture:

    Derive a class from ::testing::Test . Start its body with protected:, as we'll want to access fixture members from sub-classes.
    Inside the class, declare any objects you plan to use.
    If necessary, write a default constructor or SetUp() function to prepare the objects for each test. A common mistake is to spell SetUp() as Setup() with a small u - Use override in C++11 to make sure you spelled it correctly.
    If necessary, write a destructor or TearDown() function to release any resources you allocated in SetUp() . To learn when you should use the constructor/destructor and when you should use SetUp()/TearDown(), read the FAQ.
    If needed, define subroutines for your tests to share.
    ```

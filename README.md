# My Exes
#### It's not a package about finding your exes!

Developing a deep learning model usually includes running different **experiments** and you want to keep track of them. So, you need to log all the parameters and losses, etc and save them somewhere.  
This is where `MyExes` comes in handy! It's a simple package that provides **`MyEx`** class which can be used for logging experiments. For each experiments you need to make a `yaml` file containing the experiment name and its parameters. Then you can make an instance of `MyEx` with this yaml file as input. You can then use it to log any data you want.  
> [!NOTE]
> **My Exes** doesn't provide any automated logging mechanism but provides a simple interface for logging data manually.

### Installation
You can install my exes using pip:
```bash
pip install my-exes
```

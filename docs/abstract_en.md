<!--
 * @Author: hibana2077 hibana2077@gmail.com
 * @Date: 2024-06-14 18:15:31
 * @LastEditors: hibana2077 hibana2077@gmail.com
 * @LastEditTime: 2024-06-14 18:29:10
 * @FilePath: \Dataset-Cartography-for-Tree-based-model\docs\abstract_en.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# Abstract

In this study, we explore the possibility of applying Data Cartography, a technique originally designed for classification tasks in large language models, to tree-based models. We innovatively adapted this technique for use with decision trees and random forests, common in machine learning. This transition represents a significant innovation and contribution, particularly in the applicability and efficiency of the model, providing a new method for data analysis and processing in tree-based models.

Our research process begins with establishing a foundational model based on Data Cartography, opting for a simpler model structure than the neural networks used in the original studies to better suit the characteristics of tree-based models. We then use this foundational model to segment the dataset into three levels of difficulty: hard, medium, and easy for the model to learn. This classification not only enhances the efficiency of model training but also allows us to select the most appropriate modeling strategy for data of varying difficulties.

Following the segmentation, we utilize the model from the first step as the base model for further modeling of the segmented datasets. In this process, we particularly emphasize the use of model stacking techniques to enhance the overall performance of the models. Ultimately, our results demonstrate that this approach significantly improves the predictive capability of tree-based models on complex datasets, affirming the innovative value and practicality of Data Cartography in the application to tree-based models.
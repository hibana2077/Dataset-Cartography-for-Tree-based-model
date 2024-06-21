<!--
 * @Author: hibana2077 hibana2077@gmail.com
 * @Date: 2024-06-21 10:37:03
 * @LastEditors: hibana2077 hibana2077@gmail.com
 * @LastEditTime: 2024-06-21 11:17:54
 * @FilePath: \Dataset-Cartography-for-Tree-based-model\docs\related_work.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# Related Work

Data Cartography is a method that maps data points according to their difficulty levels during training, aiming to improve model performance by focusing on the most informative instances. Originally developed for text classification in neural networks, its adaptation to tree-based models such as decision trees and random forests represents a novel approach.

1. **Cartography Active Learning (CAL)**:
   - CAL utilizes the behavior of a model on individual instances during training to identify the most informative instances for labeling. This method has shown to be competitive with traditional active learning strategies in text classification tasks by leveraging data maps to derive insights into dataset quality [(Zhang & Plank, 2021)](https://consensus.app/papers/cartography-active-learning-zhang/d7e62d8c35215cbda66edd8042ea4e82/?utm_source=chatgpt).

2. **Machine Learning for Predictive Modeling**:
   - CART models have been widely used across different domains, such as predicting innovation efforts in firms. This approach has been found to outperform linear models in terms of accuracy, emphasizing the versatility and robustness of CART in various predictive tasks [(Rani et al., 2023)](https://consensus.app/papers/machine-learning-model-predicting-effort-firms-rani/7762df2cbd4a5720a9446e960657021b/?utm_source=chatgpt).

3. **Hybrid Models**:
   - Combining CART with other models, such as fuzzy ARTMAP neural networks, has led to hybrid models that can learn data stably while providing interpretable decision rules. This hybrid approach has proven effective in medical data classification tasks, highlighting the potential for enhancing CART with additional learning paradigms [(Seera et al., 2015)](https://consensus.app/papers/fam–cart-model-application-data-classification-seera/135e75eea34c535e858f99888489815b/?utm_source=chatgpt).

4. **Runtime Optimizations for Tree-Based Machine Learning Models**:
   - Optimizing the runtime performance of tree-based models, such as gradient-boosted regression trees, has been shown to significantly improve their efficiency. Techniques such as cache-conscious data structuring and micro-batching predictions enhance the speed and utility of these models [(Asadi et al., 2014)](https://consensus.app/papers/optimizations-treebased-machine-learning-models-asadi/5bf2a386dbcb5790bcb4aeac49a66950/?utm_source=chatgpt).

5. **Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics**

    1. **Data Maps as Diagnostic Tools**: Data Cartography utilizes "Data Maps" as a model-based tool to assess and characterize large datasets, focusing on the training dynamics of the model.

    2. **Model-Dependent Measures**: The approach introduces two specific measures for assessing data quality based on model behavior during training: the model’s confidence in the true class of each instance, and the variability of this confidence across different training epochs.

    3. **Identification of Data Regions**: Through the use of Data Maps, three distinct regions within the dataset are identified, each with specific characteristics that impact model training and performance:
        - Ambiguous regions, which contribute significantly to out-of-distribution generalization.
        - Easy-to-learn regions, which are populous and crucial for model optimization.
        - Hard-to-learn regions, which often indicate potential labeling errors.

    4. **Focus Shift Recommendation**: The results from the use of Data Maps suggest a shift from prioritizing data quantity to enhancing data quality, which could lead to the development of more robust models and improved generalization capabilities outside of the training distribution.

    5. **Practical Application and Implications**: This method of mapping and diagnosing datasets can provide valuable insights into the dynamics of model training and offer a more nuanced understanding of data quality and its impact on machine learning models.

## Conclusion

The adaptation of Data Cartography to tree-based models is supported by a variety of studies demonstrating the flexibility and efficacy of CART in different domains. This approach leverages the strengths of decision trees and random forests in handling complex datasets and provides a novel method for enhancing model performance through data segmentation and focused learning.
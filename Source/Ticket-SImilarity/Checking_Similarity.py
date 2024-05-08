import pandas as pd
from tabulate import tabulate


# --------------------------- Create new dataset with examples of each category ----------------------------------------

# Assuming you have a dataset named 'original_dataset.csv' with a column 'category'
original_dataset = pd.read_csv('../../Dataset/Cleaned_Dataset.csv')

# Initialize an empty DataFrame to store the new dataset
new_dataset = pd.DataFrame(columns=original_dataset.columns)

# Choose how many examples of each category you want
examples_per_category = 5

# Loop through each unique category
for category in original_dataset['Topic_category'].unique():
    # Filter the original dataset for examples of the current category
    category_examples = original_dataset[original_dataset['Topic_category'] == category].head(examples_per_category)
    # Append the selected examples to the new dataset
    new_dataset = new_dataset.append(category_examples)

print(tabulate(new_dataset.head(), headers='keys', tablefmt='pretty'))

# ----------------------------------- Add solution examples to the new dataset ----------------------------------------

new_dataset['Resolution'] = ["Good morning, xxxx xxxx. I understand your frustration with Chase Bank's cardmember services. When disputing a debt, it's your right as a consumer to request validation of that debt, and receiving a mere statement doesn't suffice."
                             "To address this issue, I recommend taking the following steps: Send a Certified Letter: Draft a certified letter to Chase Bank reiterating your request for debt validation. Clearly state that you require proper validation according to the Fair Debt Collection Practices Act (FDCPA). "
                             "Document Everything: Keep records of all communications with Chase, including copies of letters sent, certified mail receipts, and any responses received."
                             "Seek Legal Assistance: If Chase continues to harass you for payment without providing proper validation, consider seeking legal assistance. There are consumer protection laws in place to safeguard your rights in these situations."
                             "File a Complaint: If necessary, file a complaint with the Consumer Financial Protection Bureau (CFPB) or your state's attorney general's office. They can investigate the matter and take appropriate action if Chase is found to be in violation of debt collection laws."
                             "Remember, you have rights as a consumer, and Chase Bank must adhere to the law when attempting to collect a debt. Don't hesitate to assert those rights and seek resolution to this matter.",
                             "I'm sorry to hear about your situation. Here's a concise approach to address it:"
                             "Contact Chase: Call Chase Bank and explain the situation regarding the deposited check and the closure of your account. Provide them with the check number and claim number for reference"
                             "Clarify Ownership: Explain that your grandson is often out of the country and that you're handling the matter. Request clarification on why they closed your account and withheld the funds."
                             "Document Everything: Keep records of your conversations with Chase, including dates, times, and the names of any representatives you speak with."
                             "Consider Legal Help: If Chase doesn't resolve the issue satisfactorily, seek legal assistance. They can advise you on your rights and options for recovering the withheld funds."
                             "It's important to act promptly and persistently to resolve this matter and ensure you receive the money owed to you.",
                             "I understand the frustration and inconvenience you've experienced with Chase Bank's handling of your account. Here's how you can address it:"
                             "Contact Chase Immediately: Call Chase Bank to express your concerns and inquire about the status of your funds. Provide them with all relevant details, including dates of communication and promises made by bank representatives."
                             "Request Expedited Resolution: Politely request that Chase expedite the process of returning your funds to mitigate the financial strain you're facing due to their actions."
                             "Document Everything: Keep a record of all interactions with Chase, including dates, times, names of representatives spoken to, and any promises or commitments made by the bank."
                             "Consider Escalation: If you're unable to resolve the issue satisfactorily through regular customer service channels, consider escalating your complaint to higher levels within the bank or seeking assistance from regulatory agencies."
                             "Seek Financial Assistance: In the meantime, explore options for financial assistance to cover any immediate expenses resulting from the delayed access to your funds, such as speaking with your landlord or creditors about your situation."
                             "Remember to remain persistent and assertive in seeking resolution, as you have the right to access your funds and be treated professionally by your bank.",
                             "It sounds like you've had a frustrating experience with Chase Auto. Here's how you can address the issue:  Contact Chase Auto's customer service to address the mishandling of your account."
                             "Provide specific details and gather any relevant documentation."
                             "Clearly state your desired resolution."
                             "Consider escalation if needed, including higher levels within Chase Auto or consumer protection agencies."
                             "Seek legal advice if significant financial harm has occurred. "
                             "Remember to stay persistent and advocate for yourself throughout this process. You deserve to have your concerns addressed and your account handled appropriately.",
                             "It sounds like you're dealing with a frustrating situation with Chase Bank. Let's see how we can help you tackle it:"
                             "Contact Chase Bank's customer service to address the issue directly."
                             "Provide specific details and any relevant documentation."
                             "Clearly state your desired resolution."
                             "Consider escalating the matter if needed."
                             "Seek legal advice if the issue persists or escalates further.",
                             "It seems you've encountered an issue with your account upgrade with Chase. Here's a way to address it:"
                             "Contact Chase's customer service to discuss the discrepancy in your account's anniversary date."
                             "Provide details of the conversation with the agent and mention the recorded call."
                             "Request to have your anniversary date reverted to its original one."
                             "If necessary, escalate the matter until it's resolved to your satisfaction.",
                             "It sounds like you've had a frustrating experience with your Chase Amazon card being declined during a critical time. Here's how you can address it:"
                             "Contact Chase's customer service immediately to explain the situation."
                             "Provide specific details of the declined transactions and the assurances you received."
                             "Request a thorough investigation into the issue to ensure it doesn't happen again."
                             "Consider requesting compensation for the inconvenience caused during such a sensitive time."
                             "If the issue persists, escalate your complaint within Chase and consider reaching out to consumer protection agencies for assistance."
                             "I hope this helps, and I'm sorry for the difficulties you've faced.",
                             "It sounds like you've been through a frustrating ordeal with your Chase Amazon card being declined during a sensitive time. Here's a concise plan to address it:"
                             "Contact Chase immediately to explain the situation."
                             "Detail the declined transactions and assurances received from Chase."
                             "Request a thorough investigation and resolution to prevent future occurrences."
                             "Consider seeking compensation for the inconvenience."
                             "Escalate your complaint within Chase if needed, and seek assistance from consumer protection agencies if the issue persists."
                             "I hope this helps, and I'm sorry for the difficulties you've faced.",
                             "It's disheartening to hear about your experience with time-share companies. Here's a concise approach to address it:"
                             "Gather all documentation related to the transactions."
                             "Contact your credit card company to dispute the charges."
                             "Consider seeking assistance from consumer protection agencies or legal counsel."
                             "Be persistent in pursuing refunds and resolution from the companies involved."
                             "I hope this helps, and I'm sorry you've had to deal with this situation.",
                             "I'm sorry to hear about the breach of trust with your roommate. Here's a brief guide on how to address the situation:"
                             "Immediately contact Chase to report the unauthorized transactions and request a freeze on your account."
                             "Change your PIN and passwords for all accounts and devices."
                             "Consider filing a police report for identity theft and theft of your property."
                             "Review your account statements carefully for any further unauthorized activity."
                             "Consider finding a new living arrangement to ensure your security and peace of mind."
                             "I hope this helps, and I'm sorry you're dealing with this situation.",
                             "It sounds like you've been experiencing a frustrating issue with accessing your Chase Ultimate Rewards points. Here's what you can do to address it:"
                             "Contact Chase customer support again to explain the ongoing problem with accessing your Ultimate Rewards account."
                             "Provide all necessary details and documentation regarding your points and the error message you're encountering."
                             "Request that they escalate the issue to their technical team for further investigation."
                             "Consider reaching out to the VP of customer relations again or escalating the matter within Chase if necessary."
                             "Stress the urgency of resolving the issue due to your frequent travel and reliance on these points."
                             "I hope this helps, and I'm sorry for the inconvenience you've faced.",
                             "It sounds like you're dealing with identity theft and fraudulent applications for Chase credit cards. Here's what you can do:"
                             "1. Immediately contact Chase to report the fraudulent activity on your account."
                             "2. Request that they freeze or close the fraudulent account and provide you with a fraud report or case number."
                             "3. Consider placing a fraud alert or credit freeze on your credit reports with the major credit bureaus."
                             "4. Monitor your accounts and credit reports closely for any further suspicious activity."
                             "5. File a report with the Federal Trade Commission (FTC) and consider filing a police report for identity theft."
                             "It's important to act quickly to mitigate any potential damage to your credit and finances.",
                             "It seems you encountered an issue with a credit card application and subsequent account opening without your consent. Here's how to address it:"
                             "1. Contact Chase Bank immediately to dispute the unauthorized credit card account opened in your name."
                             "2. Provide detailed information about the situation, including the date of application and your explicit request to withdraw it."
                             "3. Request that Chase correct the information on your credit report and remove any negative impacts resulting from this error."
                             "4. If Chase refuses to cooperate, file a dispute with the credit bureaus and consider seeking legal advice or assistance from consumer protection agencies."
                             "It's crucial to resolve this issue promptly to protect your credit and financial well-being.",
                             "It seems you're requesting to remove an inquiry from your credit report. Here's how you can address it:"
                             "Contact the institution that made the inquiry and ask them to remove it."
                             "If the institution refuses, you can dispute the inquiry with the credit bureaus."
                             "Provide any supporting documentation to prove that the inquiry was unauthorized or incorrect."
                             "Monitor your credit report to ensure that the inquiry is removed."
                             "It's important to act promptly to ensure the accuracy of your credit report.",
                             "It appears there's an issue with your Chase credit card incorrectly reporting data on your credit report. Here's what you can do:"
                             "Contact Chase to explain the inaccuracies and request corrections."
                             "Provide any evidence or documentation supporting the correct information."
                             "If Chase doesn't resolve the issue, file a dispute with the credit bureaus."
                             "Follow up with both Chase and the credit bureaus until the issue is resolved."
                             "Ensuring the accuracy of your credit report is crucial, so be persistent in seeking resolution.",
                             "It seems you've encountered difficulty in obtaining a one-month payment extension for your auto loan with Chase. Here's what you can do:"
                             "Contact Chase again to request clarification on the reason for the denial and inquire about any policy changes."
                             "Keep records of all communication and documentation related to your request."
                             "If you're unsatisfied with Chase's response, consider filing a complaint with the Consumer Financial Protection Bureau (CFPB)."
                             "Provide the CFPB with all relevant details and documentation for them to investigate the denial of your payment extension request."
                             "It's important to advocate for yourself and seek resolution to this matter.",
                             "It sounds like you're facing challenges with getting a loan modification from Chase. Here's what you can do:"
                             "Keep detailed records of all communication with Chase regarding your loan modification request."
                             "Follow up regularly to ensure they have all necessary documentation and information."
                             "If Chase continues to give you the runaround or asks for unnecessary documents, consider seeking assistance from a housing counselor or legal advisor."
                             "If necessary, file a complaint with the Consumer Financial Protection Bureau (CFPB) or your state's attorney general's office."
                             "Persistence and seeking outside assistance may help you navigate this process more effectively.",
                             "It seems there's been confusion regarding the allocation of your payments with Chase. Here's what you can do:"
                             "Gather all relevant statements and documentation showing your payments and balances."
                             "Contact Chase to discuss the discrepancy and request clarification on how payments are applied."
                             "If Chase doesn't provide a satisfactory resolution, consider filing a complaint with the Consumer Financial Protection Bureau (CFPB) or seeking legal advice."
                             "Clearly outline your concerns and provide evidence to support your case."
                             "Ensuring accurate and fair treatment by financial institutions is essential, so don't hesitate to advocate for yourself.",
                             "It seems you're facing challenges with a foreclosure proceeding involving Chase Bank. Here's what you can do:"
                             "Document all communication and transactions related to the wired payment and foreclosure proceedings "
                             "Contact Chase Bank, specifically mentioning the wired payment and the discrepancy in the total payoff amount."
                             "If Chase doesn't cooperate, consider seeking legal advice or contacting a housing counselor for assistance."
                             "Provide all necessary information and documentation to support your case and advocate for a fair resolution."
                             "If needed, escalate the matter through formal channels such as filing a complaint with regulatory authorities or seeking legal action."
                             "It's essential to protect your rights and property, so don't hesitate to pursue all available avenues for resolution.",
                             "It appears you're experiencing difficulty obtaining the title for your car from Chase Financial. Here's what you can do:"
                             "1. Gather all documentation related to your car purchase and loan payments."
                             "2. Contact Chase Financial again, emphasizing the urgency of the situation and the need for them to release the title promptly."
                             "3. If Chase Financial fails to resolve the issue, consider reaching out to regulatory authorities or legal assistance."
                             "4. Explore options for obtaining temporary registration or alternative transportation while you work to resolve the title issue."
                             "It's essential to assertively pursue resolution to ensure you can legally use your vehicle.",
                             "It seems you've encountered a fraudulent transaction issue with Chase Quick Pay. Here's what you can do:"
                             "1. Contact Chase immediately to report the fraudulent transaction and request assistance in resolving the issue."
                             "2. Provide all relevant details and documentation regarding the transaction and the scam website."
                             "3. Follow Chase's instructions regarding next steps, such as contacting the recipient bank or initiating a claim."
                             "4. If you're dissatisfied with Chase's response, consider filing a complaint with regulatory authorities or seeking legal advice."
                             "5. Be vigilant in monitoring your accounts and transactions to prevent further unauthorized activity."
                             "It's essential to act swiftly to mitigate any potential losses and protect your financial security.",
                             "It appears you've been unfairly charged overdraft fees by Chase Bank. Here's what you can do:"
                             "1. Document all instances of overdraft fees and the circumstances surrounding them."
                             "2. Contact Chase Bank to dispute the fees and request a refund, providing evidence of your account activity and your attempts to rectify any low balances promptly."
                             "3. If Chase refuses to refund the fees or offers an unsatisfactory resolution, consider escalating the issue by filing a complaint with regulatory authorities or seeking legal advice."
                             "4. Be persistent in advocating for yourself and your rights as a consumer, and keep records of all communication with Chase regarding this matter."
                             "Unfairly charging overdraft fees can have significant financial implications, so it's essential to address the issue promptly and assertively.",
                             "It sounds like you've encountered an issue with an undelivered furniture purchase and a denied claim by your bank. Here's what you can do:"
                             "1. Gather all documentation related to the purchase, including receipts, order confirmations, and any correspondence with the furniture company."
                             "2. Contact the furniture company to inquire about the status of your order and request a refund."
                             "3. If the company is unresponsive or unwilling to provide a refund, escalate the issue with your bank."
                             "4. Provide your bank with evidence of the failed transaction and any attempts you've made to resolve the issue with the furniture company."
                             "5. If necessary, file a dispute with your bank or consider seeking assistance from consumer protection agencies."
                             "It's essential to act promptly to resolve the situation and recover your funds.",
                             "It appears you've experienced an unfortunate situation involving a duplicate charge, a closed account, and allegations of fraudulent activity with Chase Bank. Here's what you can do:"
                             "1. Contact Chase Bank again to clarify the situation and request an explanation for the account closure and the allegations made against you."
                             "2. Provide any evidence or documentation you have related to the duplicate charge and your attempts to resolve the issue."
                             "3. Express your concerns about the impact on your reputation and request that Chase refrain from reporting the account closure as involuntary."
                             "4. If necessary, escalate your complaint within Chase Bank or consider seeking assistance from consumer protection agencies."
                             "5. Monitor your credit report to ensure accurate reporting and dispute any inaccuracies that may arise from this incident.It's important to address this matter promptly to protect your reputation and financial well-being.",
                             "It seems you have specific concerns regarding your interactions with JP Morgan Chase Bank. Here's how you can address them:"
                             "1. **Regarding Access to Personal Information:** Request clarification from JP Morgan Chase about why they accessed your personal information through xxxx xxxx xxxx. Seek a valid reason for this action to understand the purpose behind it."
                             "2. **Discrepancy in Routing Numbers:** Inquire why there is a mismatch between the routing numbers associated with your Chase checking account. Provide evidence of the differing routing numbers and ask for an explanation from Chase regarding this inconsistency."
                             "3. **Disputed Information on Credit Report:** Follow up with JP Morgan Chase to verify if the disputed information submitted to credit bureaus has been indeed deleted due to being incomplete, inaccurate, or unverifiable. Request confirmation of the resolution of the dispute."
                             "Make sure to document all communication and provide relevant evidence to support your inquiries. If necessary, consider escalating your concerns within the bank or seeking assistance from consumer protection agencies like the Consumer Financial Protection Bureau (CFPB)."]


# Save the new dataset to a new CSV file
# new_dataset.to_csv('../../Dataset/new_dataset.csv', index=False)

print(tabulate(new_dataset.head(), headers='keys', tablefmt='pretty'))



